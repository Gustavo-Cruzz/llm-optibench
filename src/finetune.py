import logging
import torch
from omegaconf import DictConfig
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from src.dataset_handlers import DatasetHandler

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
except ImportError:
    LoraConfig, get_peft_model, prepare_model_for_kbit_training, SFTTrainer = None, None, None, None

logger = logging.getLogger(__name__)

class Finetuner:
    """
    Handles LoRA Fine-Tuning usually intended as a recovery step after pruning.
    """
    def __init__(self, cfg: DictConfig, tokenizer: AutoTokenizer, handler: DatasetHandler):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.handler = handler

    def train(self, model: AutoModelForCausalLM, dataset: Dataset) -> AutoModelForCausalLM:
        if SFTTrainer is None:
            raise ImportError("Please install `peft` and `trl` to use fine-tuning.")
            
        logger.info("Setting up LoRA adapters for recovery fine-tuning...")
        
        # Freezing the model for PEFT
        model.train()
        
        # Prepare for int4/int8 training if quantized
        if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
            model = prepare_model_for_kbit_training(model)
            
        lora_config = LoraConfig(
            r=self.cfg.optimization.get("lora_r", 8),
            lora_alpha=self.cfg.optimization.get("lora_alpha", 16),
            lora_dropout=self.cfg.optimization.get("lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()

        # We will format the text column the SFTTrainer expects
        def formatting_func(example):
            return self.handler.format_prompt(example)
            
        args = TrainingArguments(
            output_dir=self.cfg.paths.get("output_dir", "results") + "/lora",
            per_device_train_batch_size=self.cfg.optimization.get("finetune_batch_size", 2),
            learning_rate=self.cfg.optimization.get("finetune_learning_rate", 2e-4),
            num_train_epochs=self.cfg.optimization.get("finetune_epochs", 1),
            logging_steps=10,
            save_strategy="no", # Avoid dumping checkpoints for benchmark
            optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_hf",
            fp16=True,
            remove_unused_columns=False,
            report_to="none" # We handle MLflow outside
        )

        trainer = SFTTrainer(
            model=peft_model,
            train_dataset=dataset,
            max_seq_length=self.cfg.model.max_length,
            tokenizer=self.tokenizer,
            args=args,
            formatting_func=formatting_func,
        )

        logger.info("Starting Fine-tuning...")
        trainer.train()
        
        # Return the finetuned model
        return peft_model
