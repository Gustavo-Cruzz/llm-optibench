import pytest
import sys
import os
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import QADataloader
from src.model_engine import ModelOptimizer
from src.evaluation import Evaluator

@pytest.fixture
def mock_cfg():
    conf = OmegaConf.create({
        "model": {
            "name": "gpt2", # Use small model for testing
            "device": "cpu",
            "max_length": 50
        },
        "experiment": {
            "batch_size": 1,
            "dataset_subset_size": 2,
            "seed": 42
        },
        "optimization": {
            "load_in_4bit": False, # CPU doesn't support 4bit bitsandbytes usually
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "float16",
            "pruning_amount": 0.2,
            "pruning_method": "l1_unstructured"
        },
        "paths": {
            "output_dir": "results",
            "data_dir": "data"
        }
    })
    return conf

def test_dataloader(mock_cfg):
    loader = QADataloader(mock_cfg)
    # Mock load_dataset to avoid downloading SQuAD
    with patch("src.data_loader.load_dataset") as mock_load:
        mock_data = [
            {"context": "ctx", "question": "q", "answers": {"text": ["a"]}},
            {"context": "ctx2", "question": "q2", "answers": {"text": ["a2"]}}
        ]
        # Mock dataset object behavior
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 2
        mock_dataset.select.return_value = mock_data # Simplify
        mock_load.return_value = mock_dataset
        
        data = loader.load_data()
        assert data is not None

def test_model_optimizer_loading(mock_cfg):
    # Test loading baseline (CPU)
    optimizer = ModelOptimizer(mock_cfg)
    model = optimizer.load_baseline()
    assert model is not None
    assert model.config.model_type == "gpt2"

def test_pruning(mock_cfg):
    optimizer = ModelOptimizer(mock_cfg)
    model = optimizer.load_baseline()
    # Check linear layer weight before pruning (checking non-zero count is hard without specific values, 
    # but we can check if pruning hook is registered or just run it without error)
    
    model = optimizer.apply_pruning(model)
    # Just ensure it runs without error
    assert model is not None

def test_evaluator(mock_cfg):
    optimizer = ModelOptimizer(mock_cfg)
    model = optimizer.load_baseline()
    tokenizer = optimizer.tokenizer
    evaluator = Evaluator(mock_cfg, tokenizer)
    
    # Mock dataset
    dataset = [
        {"context": "London is the capital of UK.", "question": "What is the capital?", "answers": {"text": ["London"]}}
    ]
    
    metrics = evaluator.run_inference(model, dataset, "Test Run")
    assert "F1 Score" in metrics
    assert "Avg Latency (tok/s)" in metrics
