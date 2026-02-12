import logging
from typing import Any, Dict, List, Optional
from datasets import load_dataset, Dataset
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

class QADataloader:
    """
    Handles loading and formatting of the SQuAD v2 dataset for LLM evaluation.
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.dataset_name = "squad_v2"
        self.split = "validation"
    
    def load_data(self) -> Dataset:
        """
        Loads the SQuAD v2 validation dataset.
        If a subset size is defined in config, slices the dataset.
        """
        logger.info(f"Loading {self.dataset_name} ({self.split})...")
        try:
            dataset = load_dataset(self.dataset_name, split=self.split)
            
            subset_size = self.cfg.experiment.get("dataset_subset_size", None)
            if subset_size:
                logger.info(f"Slicing dataset to first {subset_size} examples.")
                dataset = dataset.select(range(min(len(dataset), subset_size)))
                
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        Wraps the input context and question into a prompt template suitable for 
        instruction-tuned models like Mistral or Llama-2.
        
        Args:
            example: A single dictionary from the dataset containing 'context' and 'question'.
            
        Returns:
            Formatted prompt string.
        """
        # Generic instruction template. 
        # For strict chat models ensuring specific chat templates (like [INST]) is crucial.
        # This implementation assumes a standard prompt format often used in benchmarks.
        
        context = example["context"]
        question = example["question"]
        
        # Mistral / Llama-2 style instruction format
        prompt = f"[INST] Read the following context and answer the question.\n\nContext: {context}\n\nQuestion: {question} [/INST]"
        
        return prompt

    def get_ground_truths(self, example: Dict[str, Any]) -> List[str]:
        """
        Extracts valid answers from the example.
        SQuAD v2 has 'answers' key with 'text' list.
        """
        return example["answers"]["text"] if example["answers"]["text"] else []
