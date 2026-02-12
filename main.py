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
import gc
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from src.data_loader import QADataloader
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
    if args.batch is not None:
        base.experiment.batch_size = args.batch
    if args.pruning is not None:
        base.optimization.pruning_amount = args.pruning
    if args.device is not None:
        base.model.device = args.device

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
    data_loader = QADataloader(cfg)
    dataset = data_loader.load_data()

    results: list[dict] = []

    # ── RUN 1: BASELINE (FP16) ──────────────────────────────────────────────
    logger.info("=== RUN 1: BASELINE (FP16) ===")
    cleanup_gpu()
    optimizer = ModelOptimizer(cfg)
    model = optimizer.load_baseline()
    evaluator = Evaluator(cfg, optimizer.tokenizer)
    metrics = evaluator.run_inference(model, dataset, "Baseline (FP16)")
    results.append(metrics)

    del model, evaluator, optimizer
    cleanup_gpu()

    # ── RUN 2: QUANTIZATION (4-bit NF4) ─────────────────────────────────────
    logger.info("=== RUN 2: QUANTIZATION (4-bit NF4) ===")
    cleanup_gpu()
    optimizer = ModelOptimizer(cfg)
    model = optimizer.apply_quantization()
    evaluator = Evaluator(cfg, optimizer.tokenizer)
    metrics = evaluator.run_inference(model, dataset, "Quantized (4-bit NF4)")
    results.append(metrics)

    del model, evaluator, optimizer
    cleanup_gpu()

    # ── RUN 3: PRUNING (Unstructured) ────────────────────────────────────────
    logger.info("=== RUN 3: PRUNING (Unstructured) ===")
    cleanup_gpu()
    optimizer = ModelOptimizer(cfg)
    model = optimizer.load_baseline()
    model = optimizer.apply_pruning(model)
    evaluator = Evaluator(cfg, optimizer.tokenizer)
    metrics = evaluator.run_inference(model, dataset, "Pruned (Unstructured)")
    results.append(metrics)

    del model, evaluator, optimizer
    cleanup_gpu()


    # ── Save Results ─────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = str(cfg.paths.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"benchmark_report_{timestamp}.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Benchmark completed. Results saved to {output_path}")
    print(df)


if __name__ == "__main__":
    main()
