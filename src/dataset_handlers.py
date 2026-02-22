from typing import Any, Dict, List, Tuple
from datasets import Dataset, load_dataset
import numpy as np
import re
import string

class DatasetHandler:
    """Abstract base class for dataset handlers."""
    def load_data(self) -> Dataset:
        raise NotImplementedError

    def format_prompt(self, example: Dict[str, Any]) -> str:
        raise NotImplementedError

    def get_references(self, example: Dict[str, Any]) -> List[Any]:
        raise NotImplementedError

    def compute_metrics(self, predictions: List[str], references: List[List[Any]]) -> Dict[str, float]:
        raise NotImplementedError

class SquadV2Handler(DatasetHandler):
    def __init__(self, split="validation", subset_size=None):
        self.dataset_name = "squad_v2"
        self.split = split
        self.subset_size = subset_size

    def load_data(self) -> Dataset:
        dataset = load_dataset(self.dataset_name, split=self.split)
        if self.subset_size:
            dataset = dataset.select(range(min(len(dataset), self.subset_size)))
        return dataset

    def format_prompt(self, example: Dict[str, Any]) -> str:
        context = example["context"]
        question = example["question"]
        return f"[INST] Read the following context and answer the question.\n\nContext: {context}\n\nQuestion: {question} [/INST]"

    def get_references(self, example: Dict[str, Any]) -> List[str]:
        return example["answers"]["text"] if example["answers"]["text"] else []

    def compute_metrics(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        f1_scores = []
        em_scores = []
        for pred, refs in zip(predictions, references):
            if not refs:
                f1_scores.append(1.0 if not pred else 0.0)
                em_scores.append(1.0 if not pred else 0.0)
                continue
            f1_scores.append(max([self._f1_score(pred, ref) for ref in refs]))
            em_scores.append(max([self._exact_match_score(pred, ref) for ref in refs]))
        return {
            "f1_score": np.mean(f1_scores) * 100,
            "exact_match": np.mean(em_scores) * 100
        }

    def _normalize_answer(self, s: str) -> str:
        def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text): return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text): return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _f1_score(self, prediction: str, ground_truth: str) -> float:
        pred_tokens = self._normalize_answer(prediction).split()
        truth_tokens = self._normalize_answer(ground_truth).split()
        common = set(pred_tokens) & set(truth_tokens)
        num_same = len(common)
        if num_same == 0: return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(truth_tokens)
        return 2 * (precision * recall) / (precision + recall)

    def _exact_match_score(self, prediction: str, ground_truth: str) -> float:
        return float(self._normalize_answer(prediction) == self._normalize_answer(ground_truth))


class Gsm8kHandler(DatasetHandler):
    def __init__(self, split="test", subset_size=None):
        self.dataset_name = "gsm8k"
        self.config_name = "main"
        self.split = split
        self.subset_size = subset_size

    def load_data(self) -> Dataset:
        dataset = load_dataset(self.dataset_name, self.config_name, split=self.split)
        if self.subset_size:
            dataset = dataset.select(range(min(len(dataset), self.subset_size)))
        return dataset

    def format_prompt(self, example: Dict[str, Any]) -> str:
        question = example["question"]
        return f"[INST] Solve the following math problem step-by-step. At the end, state your final answer as 'Final Answer: [number]'.\n\nQuestion: {question} [/INST]"

    def get_references(self, example: Dict[str, Any]) -> List[str]:
        # GSM8k answers typically end with '#### [number]'
        ans_str = example["answer"]
        match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', ans_str)
        if match:
            return [match.group(1)]
        return [ans_str.split()[-1]] # fallback

    def compute_metrics(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        em_scores = []
        for pred, refs in zip(predictions, references):
            ground_truth = refs[0].replace(',', '')
            # Try to extract the final number from the prediction
            pred_match = re.search(r'(?i)final answer.*?(?:is|:)\s*(-?\d+(?:\.\d+)?)', pred)
            if pred_match:
                extracted_pred = pred_match.group(1)
            else:
                # Fallback: find all numbers and pick the last one
                nums = re.findall(r'-?\d+(?:\.\d+)?', pred.replace(',', ''))
                extracted_pred = nums[-1] if nums else ""
                
            em_scores.append(1.0 if extracted_pred == ground_truth else 0.0)
            
        return {
            "accuracy": np.mean(em_scores) * 100
        }
