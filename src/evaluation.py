import logging
import time
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
from omegaconf import DictConfig
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

class Evaluator:
    """
    Runs the inference loop and calculates performance metrics.
    """
    def __init__(self, cfg: DictConfig, tokenizer: AutoTokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
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
            prompt = self._format_prompt(example)
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
            references.append(example["answers"]["text"] if example["answers"]["text"] else [])

        total_time = time.time() - start_time
        avg_latency = np.mean(latencies)
        
        # Calculate accuracy metrics
        f1, em = self._compute_metrics(predictions, references)
        
        # Memory usage
        peak_vram = 0
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB

        metrics = {
            "Run Name": run_name,
            "F1 Score": f1,
            "Exact Match (EM)": em,
            "Avg Latency (tok/s)": avg_latency,
            "Total Time (s)": total_time,
            "Peak VRAM (GB)": peak_vram,
            "Model Size (GB)": self._get_model_size(model) 
        }
        
        logger.info(f"Results for {run_name}: {metrics}")
        return metrics

    def _format_prompt(self, example: Dict[str, Any]) -> str:
        # Reusing the simple prompt strategy, or could inject the dataloader's formatter
        context = example["context"]
        question = example["question"]
        return f"[INST] Read the following context and answer the question.\n\nContext: {context}\n\nQuestion: {question} [/INST]"

    def _compute_metrics(self, predictions: List[str], references: List[List[str]]) -> Tuple[float, float]:
        """
        Computes F1 and Exact Match scores.
        """
        f1_scores = []
        em_scores = []
        
        for pred, refs in zip(predictions, references):
            if not refs:
                # If no answer (unanswerable), check if prediction is empty
                f1_scores.append(1.0 if not pred else 0.0)
                em_scores.append(1.0 if not pred else 0.0)
                continue
                
            # Compute max F1/EM over all valid references
            ce_f1 = [self._f1_score(pred, ref) for ref in refs]
            ce_em = [self._exact_match_score(pred, ref) for ref in refs]
            
            f1_scores.append(max(ce_f1))
            em_scores.append(max(ce_em))
            
        return np.mean(f1_scores) * 100, np.mean(em_scores) * 100

    def _f1_score(self, prediction: str, ground_truth: str) -> float:
        pred_tokens = self._normalize_answer(prediction).split()
        truth_tokens = self._normalize_answer(ground_truth).split()
        
        common = set(pred_tokens) & set(truth_tokens)
        num_same = len(common)
        
        if num_same == 0:
            return 0.0
        
        precision = num_same / len(pred_tokens)
        recall = num_same / len(truth_tokens)
        
        return 2 * (precision * recall) / (precision + recall)

    def _exact_match_score(self, prediction: str, ground_truth: str) -> float:
        return float(self._normalize_answer(prediction) == self._normalize_answer(ground_truth))

    def _normalize_answer(self, s: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""
        import string, re
        
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

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
