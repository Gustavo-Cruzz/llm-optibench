import torch
import torch.nn as nn
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

@torch.no_grad()
def apply_wanda_pruning(model, dataloader, tokenizer, device, sparsity_ratio, calibration_samples=128):
    """
    Applies the Wanda (Pruning by Weights and activations) algorithm.
    Wanda decides pruning thresholds based on the product of weight magnitudes
    and the input activation norms (||X||_2).
    """
    logger.info(f"Starting Wanda Pruning calibration with {calibration_samples} samples...")
    
    # 1. Identify all linear layers
    # We maintain a dictionary to store the L2 norm of the input activations
    activation_norms = {}
    hooks = []
    
    def get_activation_hook(name):
        def hook(module, input, output):
            # input[0] shape info: (batch, seq_len, hidden_dim)
            x = input[0]
            if x.dim() == 3:
                x = x.view(-1, x.shape[-1]) # flatten batch and seq_len
            
            # calculate L2 norm along the feature dimension for each input feature
            # ||X||_2 for each column (feature)
            norm = torch.norm(x, p=2, dim=0)
            
            if name not in activation_norms:
                activation_norms[name] = norm
            else:
                # Accumulate moving average or sum, here we just sum and normalize later if needed,
                # but Wanda technically just uses a scaling factor so sum is proportional.
                activation_norms[name] += norm
                
        return hook

    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            hooks.append(module.register_forward_hook(get_activation_hook(name)))
            
    # 2. Forward pass with calibration data
    model.eval()
    samples_processed = 0
    
    for example in dataloader:
        if samples_processed >= calibration_samples:
            break
            
        # Re-use the existing prompt format from Evaluator, or assume text context
        context = example["context"]
        question = example["question"]
        prompt = f"[INST] Read the following context and answer the question.\n\nContext: {context}\n\nQuestion: {question} [/INST]"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # We only need the forward pass
        model(**inputs)
        samples_processed += 1

    # Remove hooks
    for hook in hooks:
        hook.remove()

    logger.info(f"Calibration complete. Applying sparsity mask (Ratio: {sparsity_ratio})...")

    # 3. Compute Wanda metric and apply mask
    pruned_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            if name in activation_norms:
                W = module.weight.data
                X_norm = activation_norms[name].unsqueeze(0) # (1, in_features)
                
                # Wanda metric: S = |W| * ||X||_2
                S = torch.abs(W) * X_norm
                
                # Flatten S locally per layer and compute threshold 
                sort_res, _ = torch.sort(S, dim=-1, descending=False)
                threshold_idx = int(W.shape[1] * sparsity_ratio)
                
                # Handling boundary condition
                if threshold_idx == W.shape[1]:
                    threshold_idx -= 1
                    
                thresholds = sort_res[:, threshold_idx].unsqueeze(1)
                
                # Create mask
                mask = (S >= thresholds).float()
                
                # Apply mask to weights
                module.weight.data.mul_(mask)
                pruned_count += 1
                
    logger.info(f"Wanda pruning applied successfully to {pruned_count} layers.")
    return model
