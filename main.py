"""
LLM OptiBench — Main entry point.

Replaces Hydra with plain YAML + OmegaConf to avoid Python 3.14 compatibility issues.
Usage:
    python main.py                    # default (TinyLlama)
    python main.py --model phi2       # use Phi-2
    python main.py --model mistral    # use Mistral-7B
    python main.py --subset 50        # evaluate on 50 samples
    python main.py --pruning 0.3      # 30% sparsity
"""

import argparse
from contextlib import nullcontext
import gc
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml
import mlflow
from omegaconf import DictConfig, OmegaConf

from src.data_loader import get_dataset_handler
from src.evaluation import Evaluator
from src.model_engine import ModelOptimizer
from src.utils import set_seed, setup_logging

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = ROOT_DIR / "configs"


def load_config(args: argparse.Namespace) -> DictConfig:
    """Load base YAML config, merge model-specific YAML, then apply CLI overrides."""

    # 1. Load base config
    base_path = CONFIG_DIR / "config.yaml"
    with open(base_path, "r") as f:
        base = OmegaConf.create(yaml.safe_load(f))

    # 2. If a model preset was chosen, merge from configs/model/<name>.yaml
    model_yaml = CONFIG_DIR / "model" / f"{args.model}.yaml"
    if model_yaml.exists():
        with open(model_yaml, "r") as f:
            model_overrides = OmegaConf.create(yaml.safe_load(f))
        # Merge model-specific values into base.model
        base.model = OmegaConf.merge(base.model, model_overrides)
    else:
        logger.warning(
            f"No model preset '{args.model}' found at {model_yaml}. Using defaults."
        )

    # 3. Apply any CLI-level scalar overrides
    if args.subset is not None:
        base.experiment.dataset_subset_size = args.subset
    if args.dataset is not None:
        base.experiment.dataset_name = args.dataset
    if args.finetune:
        base.optimization.finetune_after_pruning = True
    if args.batch is not None:
        base.experiment.batch_size = args.batch
    if args.pruning is not None:
        base.optimization.pruning_amount = args.pruning
    if args.device is not None:
        base.model.device = args.device
    if args.export_models:
        base.paths.export_models = True

    return base


def cleanup_gpu() -> None:
    """Attempts to clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM OptiBench — Benchmark LLM optimization techniques."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tinyllama",
        help="Model preset name (matches configs/model/<name>.yaml). "
             "Choices: tinyllama, phi2, mistral  (default: tinyllama)",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Number of dataset samples to evaluate (default: from config)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size override (default: from config)",
    )
    parser.add_argument(
        "--pruning",
        type=float,
        default=None,
        help="Pruning sparsity ratio, e.g. 0.2 for 20%% (default: from config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, e.g. 'cpu' or 'cuda' (default: from config)",
    )
    parser.add_argument(
        "--export_models",
        action="store_true",
        help="Export models to disk after evaluating (default: false)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset override, e.g. 'squad_v2' or 'gsm8k' (default: from config)",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Run LoRA recovery fine-tuning after pruning (default: false)",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_config(args)
    set_seed(cfg.experiment.seed)

    logger.info(
        f"Starting LLM OptiBench with config:\n{OmegaConf.to_yaml(cfg)}"
    )

    # 1. Load Data
    handler = get_dataset_handler(cfg)
    dataset = handler.load_data()

    results: list[dict] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── MLflow Tracking Setup ────────────────────────────────────────────────
    use_mlflow = cfg.get("tracking", {}).get("use_mlflow", False)
    tracking_uri = cfg.get("tracking", {}).get("tracking_uri", "./mlruns")
    experiment_name = cfg.get("tracking", {}).get("experiment_name", "LLM_OptiBench")
    
    if use_mlflow:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    
    # ── Parent Run ───────────────────────────────────────────────────────────
    with mlflow.start_run(run_name=f"Benchmark_{args.model}_{timestamp}") if use_mlflow else nullcontext():
        if use_mlflow:
            # Log full config as artifact
            config_dict = OmegaConf.to_container(cfg, resolve=True)
            mlflow.log_dict(config_dict, "config.yaml")

            # Log key parameters
            mlflow.log_params({
                "model_name": cfg.model.name,
                "dataset_name": cfg.experiment.get("dataset_name", "squad_v2"),
                "device": cfg.model.device,
                "dataset_subset_size": cfg.experiment.get("dataset_subset_size", "all"),
                "batch_size": cfg.experiment.batch_size,
                "pruning_amount": cfg.optimization.get("pruning_amount", 0.0),
                "pruning_method": cfg.optimization.get("pruning_method", "none"),
                "finetuned_after_pruning": cfg.optimization.get("finetune_after_pruning", False),
                "quantization_method": cfg.optimization.get("quantization_method", "bitsandbytes"),
                "quant_type": cfg.optimization.get("bnb_4bit_quant_type", "none")
            })

        # ── RUN 1: BASELINE (FP16) ──────────────────────────────────────────────
        logger.info("=== RUN 1: BASELINE (FP16) ===")
        cleanup_gpu()
        optimizer = ModelOptimizer(cfg)
        model = optimizer.load_baseline()
        evaluator = Evaluator(cfg, optimizer.tokenizer, handler)
        
        with mlflow.start_run(run_name="Baseline_FP16", nested=True) if use_mlflow else nullcontext():
            metrics = evaluator.run_inference(model, dataset, "Baseline (FP16)")
            if use_mlflow:
                # Dynamically log all numeric metrics returned
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            results.append(metrics)

        if cfg.paths.get("export_models", False):
            export_path = os.path.join(cfg.paths.get("export_dir", "exports"), f"{args.model}_baseline")
            logger.info(f"Saving Baseline model to {export_path}")
            os.makedirs(export_path, exist_ok=True)
            model.save_pretrained(export_path)
            optimizer.tokenizer.save_pretrained(export_path)

        del model, evaluator, optimizer
        cleanup_gpu()

        # ── RUN 2: QUANTIZATION ─────────────────────────────────────
        logger.info(f"=== RUN 2: QUANTIZATION ({cfg.optimization.get('quantization_method', 'bitsandbytes')}) ===")
        cleanup_gpu()
        optimizer = ModelOptimizer(cfg)
        model = optimizer.apply_quantization(dataset=dataset)
        evaluator = Evaluator(cfg, optimizer.tokenizer, handler)
        
        with mlflow.start_run(run_name=f"Quantized_{cfg.optimization.get('quantization_method', 'bitsandbytes')}", nested=True) if use_mlflow else nullcontext():
            metrics = evaluator.run_inference(model, dataset, f"Quantized ({cfg.optimization.get('quantization_method', 'bitsandbytes')})")
            if use_mlflow:
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            results.append(metrics)

        if cfg.paths.get("export_models", False):
            export_path = os.path.join(cfg.paths.get("export_dir", "exports"), f"{args.model}_quantized_4bit")
            logger.info(f"Attempting to save Quantized model to {export_path}")
            os.makedirs(export_path, exist_ok=True)
            try:
                # 4-bit models in transformers might have limitations on direct save without PEFT
                model.save_pretrained(export_path)
                optimizer.tokenizer.save_pretrained(export_path)
            except Exception as e:
                logger.warning(f"Failed to save 4-bit quantized model directly: {e}")

        del model, evaluator, optimizer
        cleanup_gpu()

        # ── RUN 3: PRUNING (Unstructured) ────────────────────────────────────────
        logger.info("=== RUN 3: PRUNING (Unstructured) ===")
        cleanup_gpu()
        optimizer = ModelOptimizer(cfg)
        model = optimizer.load_baseline()
        model = optimizer.apply_pruning(model, dataset=dataset)
        
        # Optional: Recovery Fine-Tuning
        if cfg.optimization.get("finetune_after_pruning", False):
            from src.finetune import Finetuner
            logger.info("=== RECOVERY FINE-TUNING (LoRA) ===")
            finetuner = Finetuner(cfg, optimizer.tokenizer, handler)
            model = finetuner.train(model, dataset)
            
        evaluator = Evaluator(cfg, optimizer.tokenizer, handler)
        
        run_name_prefix = "Pruned_Finetuned" if cfg.optimization.get("finetune_after_pruning", False) else "Pruned_Unstructured"
        with mlflow.start_run(run_name=run_name_prefix, nested=True) if use_mlflow else nullcontext():
            display_name = "Pruned + LoRA" if cfg.optimization.get("finetune_after_pruning", False) else "Pruned (Unstructured)"
            metrics = evaluator.run_inference(model, dataset, display_name)
            if use_mlflow:
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            results.append(metrics)

        if cfg.paths.get("export_models", False):
            export_path = os.path.join(cfg.paths.get("export_dir", "exports"), f"{args.model}_pruned")
            logger.info(f"Saving Pruned model to {export_path}")
            os.makedirs(export_path, exist_ok=True)
            model.save_pretrained(export_path)
            optimizer.tokenizer.save_pretrained(export_path)

        del model, evaluator, optimizer
        cleanup_gpu()


    # ── Save Results ─────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    output_dir = str(cfg.paths.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"benchmark_report_{timestamp}.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Benchmark completed. Results saved to {output_path}")
    print(df)


if __name__ == "__main__":
    main()
