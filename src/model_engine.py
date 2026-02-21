import logging
import torch
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from omegaconf import DictConfig
from typing import Optional, Tuple
from datasets import Dataset

from src.wanda import apply_wanda_pruning

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """
    Handles model loading and optimization techniques:
    - Baseline (FP16)
    - Quantization (4-bit NF4)
    - Pruning (Unstructured Magnitude)
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model_name = cfg.model.name
        self.device = cfg.model.device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Ensure pad token is set for generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_baseline(self) -> AutoModelForCausalLM:
        """
        Loads the model in FP16 precision.
        """
        logger.info(f"Loading baseline model: {self.model_name} (FP16)")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        return model

    def apply_quantization(self) -> AutoModelForCausalLM:
        """
        Loads the model with 4-bit NF4 quantization using bitsandbytes.
        """
        logger.info(f"Loading quantized model: {self.model_name} (4-bit NF4)")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.cfg.optimization.load_in_4bit,
            bnb_4bit_quant_type=self.cfg.optimization.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, self.cfg.optimization.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=self.device,
        )
        return model

    def apply_pruning(self, model: AutoModelForCausalLM, dataset: Optional[Dataset] = None) -> AutoModelForCausalLM:
        """
        Applies either Wanda or unstructured magnitude pruning.
        
        Args:
            model: The PyTorch model to prune.
            dataset: Calibration dataset for Wanda pruning.
            
        Returns:
            Pruned model.
        """
        amount = self.cfg.optimization.pruning_amount
        method = self.cfg.optimization.pruning_method
        logger.info(f"Applying {method} pruning with amount={amount}...")

        if method == "wanda":
            if dataset is None:
                raise ValueError("Wanda pruning requires a calibration dataset.")
            calibration_samples = self.cfg.optimization.get("wanda_calibration_samples", 128)
            model = apply_wanda_pruning(
                model=model, 
                dataloader=dataset, 
                tokenizer=self.tokenizer, 
                device=self.device, 
                sparsity_ratio=amount, 
                calibration_samples=calibration_samples
            )
            return model

        # Prune only linear layers 
        parameters_to_prune = []
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Skip the output head to avoid messing up logits shape too much
                if "lm_head" in name:
                    continue
                parameters_to_prune.append((module, 'weight'))

        if not parameters_to_prune:
            logger.warning("No linear layers found to prune.")
            return model

        # Apply global unstructured pruning
        for module, name in parameters_to_prune:
            if method == "l1_unstructured":
                prune.l1_unstructured(module, name=name, amount=amount)
            elif method == "random_unstructured":
                prune.random_unstructured(module, name=name, amount=amount)
            
            # Make pruning permanent
            prune.remove(module, name)
            
        logger.info(f"Pruning complete. {len(parameters_to_prune)} layers pruned.")
        return model
