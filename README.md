# Advanced GPT-2 with Custom Positional Encodings

A PyTorch implementation of GPT-2 that explores various positional encoding methods and attention mechanisms for enhanced language modeling. This project implements multiple novel positional encoding techniques including Legendre polynomials, Chebyshev polynomials, and Gaussian encodings.

Training Report - [Link](https://api.wandb.ai/links/tulasiram/7sevdwk6)

## Features

- **Multiple Positional Encoding Methods:**
  - Learned Embeddings (default GPT-2 style)
  - Sinusoidal (Transformer style)
  - Polynomial (Legendre)
  - Polynomial (Chebyshev)
  - Gaussian
  - Random Fourier Features
  - Wavelet-based
  - Bessel Functions
  - Alternative Trigonometric

- **Attention Mechanisms:**
  - Standard Attention
  - Flash Attention v2 (when available)
  - RoPE (Rotary Position Embeddings)
  - ALiBi (Attention with Linear Biases)
  - Relative Position Attention

- **Training Features:**
  - Distributed training support (DDP)
  - Mixed precision training (FP16/BF16)
  - Gradient accumulation
  - Learning rate scheduling
  - Checkpoint saving and resuming
  - Comprehensive logging with WandB
  - Memory-efficient data loading with memory mapping

## Installation
```bash
# Clone the repository
git clone https://github.com/0-hero/gpt2-positional-encoding-experiment.git
cd gpt2-positional-encoding-experiment

# Set up the environment
bash setup.sh
```

## Data Preparation
1. Prepare the evaluation datasets:
```bash
python prepare_evaluation_data.py
```

2. Preprocess your training data:
```bash
python preprocess.py
```

## Training
Basic training command:
```bash
python train.py
```
Distributed training:
```bash
torchrun --standalone --nproc_per_node=N train.py --batch_size=96
```
Where N is the number of GPUs to use.

## Configuration

Key configuration options in `train.py`:
```python
# Model architecture
n_layer = 4
n_head = 4
n_embd = 256
block_size = 512
Training
batch_size = 12
learning_rate = 6e-4
max_iters = 10000
weight_decay = 1e-1
Positional Encodings
embedding_types = ['learned', 'sinusoidal', 'polynomial_legendre',
'polynomial_chebyshev', 'random_fourier', 'wavelet']
```

## Evaluation

The model is evaluated on multiple datasets:
- WikiText-103
- Penn Treebank (PTB)
- LAMBADA

Evaluation metrics are automatically logged to Weights & Biases during training.

## Project Structure
```
├── model.py # Core GPT model implementation
├── train.py # Training script
├── preprocess.py # Data preprocessing
├── prepare_evaluation_data.py # Evaluation data preparation
├── setup.sh # Environment setup script
└── README.md # This file
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- tiktoken
- wandb
- numpy
- scipy
- pandas
- tqdm
- huggingface_hub
