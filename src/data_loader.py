import logging
from omegaconf import DictConfig
from src.dataset_handlers import DatasetHandler, SquadV2Handler, Gsm8kHandler

logger = logging.getLogger(__name__)

def get_dataset_handler(cfg: DictConfig) -> DatasetHandler:
    """
    Factory function to return the appropriate DatasetHandler based on config.
    """
    dataset_name = cfg.experiment.get("dataset_name", "squad_v2")
    subset_size = cfg.experiment.get("dataset_subset_size", None)
    
    logger.info(f"Initializing handler for dataset: {dataset_name} (subset: {subset_size})")
    
    if dataset_name == "squad_v2":
        return SquadV2Handler(subset_size=subset_size)
    elif dataset_name == "gsm8k":
        return Gsm8kHandler(subset_size=subset_size)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
