# LLM OptiBench

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

A production-grade Python pipeline designed to benchmark Open Source LLMs (up to 7B parameters) on Question Answering tasks. This project analyzes the trade-offs between model size, inference speed, and accuracy across three optimization strategies: Baseline (FP16), Quantization (4-bit NF4), and Pruning (Unstructured).

## Features

- **Model Support**: `TinyLlama-1.1B` (default), `Phi-2`, `Mistral-7B-v0.1`.
- **Datasets Supported**:
  - `squad_v2`: Extractive Question Answering.
  - `gsm8k`: Mathematical Reasoning.
- **Optimization Techniques**:
  - **Baseline**: FP16 (Half Precision).
  - **Quantization**: 4-bit Normal Float (NF4) via QLoRA config (bitsandbytes).
  - **Pruning**: Unstructured magnitude pruning (20-30% sparsity).
- **Metrics**: F1 Score, Exact Match (EM), Latency (tok/s), Peak VRAM usage, and Model Size.

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/yourusername/llm-optibench.git
    cd llm-optibench
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

### Basic Usage

Run the full benchmark suite with default settings (TinyLlama-1.1B):

```bash
python main.py
```

### Different Configurations

You can override configuration parameters via command line arguments.

**Select a Model:**

- **TinyLlama-1.1B (Default)** - Best for 6GB VRAM:
  ```bash
  python main.py --model tinyllama
  ```
- **Phi-2 (2.7B)** - Possible on 6GB (Quantized) / 8GB+:
  ```bash
  python main.py --model phi2
  ```
- **Mistral-7B** - Requires 12GB+ VRAM:
  ```bash
  python main.py --model mistral
  ```

**Other Overrides:**

```bash
# Save models after the run
python main.py --export_models

# Run mathematical reasoning benchmark instead of QA
python main.py --dataset gsm8k

# Evaluate on 50 samples only
python main.py --subset 50

# Change pruning sparsity to 30%
python main.py --pruning 0.3

# Run on CPU
python main.py --device cpu
```

### Experiment Tracking (MLflow)

This project uses **MLflow** to track experiments, parameters, and metrics automatically. By default, runs are saved locally in the `./mlruns` directory.

To view the results in your browser:

```bash
mlflow ui
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) to compare F1 scores, latencies, and VRAM usage across the Baseline, Quantization, and Pruning runs.

### Jupyter Notebook Analysis

We also auto-generate rich Seaborn visualizations pulling directly from the MLflow API.

1. Install Jupyter (`pip install jupyter seaborn`).
2. Run `jupyter notebook notebooks/analyze_results.ipynb`.
3. View high-quality bar charts and trade-off scatter plots (F1 vs Latency vs VRAM).

## Optimization Methods

### Quantization (4-bit NF4)

We utilize `bitsandbytes` to load models in 4-bit Normal Float (NF4) precision. NF4 maps each weight to one of 16 discrete values drawn from a normal distribution, which is information-theoretically optimal for normally-distributed neural network weights (Dettmers et al., 2023). This reduces VRAM usage by approximately 75% compared to FP16.

### Pruning (Unstructured)

We apply unstructured L1 magnitude pruning to all `nn.Linear` layers (excluding `lm_head`) using `torch.nn.utils.prune`. A configurable fraction (default: 20%) of the lowest-magnitude weights are zeroed, and the pruning mask is made permanent via `prune.remove()`.

### Note on Hybrid (Quantization + Pruning) — Excluded

A fourth stage combining 4-bit quantization with unstructured pruning was evaluated during development and excluded from the final pipeline due to a fundamental incompatibility between the two techniques as currently implemented in the PyTorch/bitsandbytes ecosystem.

**Technical rationale.** The `bitsandbytes` library performs quantization at model load time, packing weight tensors into `Params4bit` objects that store weights as 4-bit integers with block-wise quantization constants. When `torch.nn.utils.prune.l1_unstructured` is subsequently applied, PyTorch's pruning API dequantizes the weights to apply a magnitude-based binary mask and then calls `prune.remove()`, which writes the masked (sparse) floating-point values back into the parameter. However, these sparse values are immediately re-quantized into the NF4 codebook, which contains only 16 discrete levels and does not include an exact zero representation. As a result, the intended zero-valued weights are mapped to the nearest non-zero NF4 level, effectively corrupting the weight distribution rather than inducing sparsity. In our experiments, this produced F1 = 0.0 and EM = 0.0 on SQuAD v2, confirming catastrophic degradation.

The reverse order (prune then quantize) is also infeasible because `bitsandbytes` only supports quantization during `AutoModelForCausalLM.from_pretrained()` and does not expose an API for post-hoc quantization of an already-loaded model.

Viable hybrid approaches would require purpose-built sparse-quantization methods such as SparseGPT (Frantar & Alistarh, 2023) or joint sparsity-aware quantization schemes (e.g., GPTQ with structured pruning), which fall outside the scope of this research.

## Project Structure

```text
llm-optibench/
├── configs/              # YAML configs (base + model presets)
│   ├── config.yaml
│   └── model/            # Model-specific overrides
├── data/                 # Local data cache
├── notebooks/            # Jupyter notebook for visualizing results
├── results/              # Output CSV benchmark reports
├── src/                  # Source code
│   ├── model_engine.py   # Model loading & optimization logic
│   ├── evaluation.py     # Inference & metric calculation
│   ├── data_loader.py    # SQuAD v2 loading & processing
│   └── utils.py          # Helper functions
├── tests/                # Unit tests
├── main.py               # Entry point
└── requirements.txt
```
