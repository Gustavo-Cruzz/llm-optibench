import logging
import time
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
from omegaconf import DictConfig
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.dataset_handlers import DatasetHandler

logger = logging.getLogger(__name__)

class Evaluator:
    """
    Runs the inference loop and calculates performance metrics.
    """
    def __init__(self, cfg: DictConfig, tokenizer: AutoTokenizer, handler: DatasetHandler):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.handler = handler
        self.device = cfg.model.device
        self.max_length = cfg.model.max_length

    def run_inference(self, model: AutoModelForCausalLM, dataset: Dataset, run_name: str) -> Dict[str, Any]:
        """
        Executes the inference loop on the provided dataset.
        
        Args:
            model: The loaded model (baseline, quantized, or pruned).
            dataset: The SQuAD v2 validation dataset.
            run_name: Identifier for the current run (e.g., "Baseline", "Quantized").
            
        Returns:
            Dictionary containing computed metrics.
        """
        logger.info(f"Starting inference run: {run_name}")
        
        model.eval()
        latencies = []
        predictions = []
        references = []
        
        # Reset peak memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        start_time = time.time()
        
        for i, example in enumerate(tqdm(dataset, desc=f"Evaluating {run_name}")):
            prompt = self.handler.format_prompt(example)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Measure generation time
            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=50, 
                    pad_token_id=self.tokenizer.eos_token_id
                )
            t1 = time.time()
            
            # Calculate latency (tokens/sec)
            generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
            latency = generated_tokens / (t1 - t0)
            latencies.append(latency)
            
            # Decode and store prediction
            pred_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            predictions.append(pred_text.strip())
            references.append(self.handler.get_references(example))

        total_time = time.time() - start_time
        avg_latency = np.mean(latencies)
        
        # Calculate accuracy metrics using the handler
        accuracy_metrics = self.handler.compute_metrics(predictions, references)
        
        # Memory usage
        peak_vram = 0
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB

        metrics = {
            "Run Name": run_name,
            **accuracy_metrics,
            "Avg Latency (tok/s)": avg_latency,
            "Total Time (s)": total_time,
            "Peak VRAM (GB)": peak_vram,
            "Model Size (GB)": self._get_model_size(model) 
        }
        
        logger.info(f"Results for {run_name}: {metrics}")
        return metrics

    def _get_model_size(self, model: AutoModelForCausalLM) -> float:
        """Estimates model size in GB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        size_all_mb = (param_size + buffer_size) / 1024**3
        return size_all_mb
