# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Gemma 3 270M fine-tuning project** using LoRA (Low-Rank Adaptation) with MLX on Apple Silicon. The project provides a complete pipeline for fine-tuning Google's Gemma 3 270M instruction-tuned model using both Alpaca (general instruction following) and SQL (text-to-SQL generation) datasets, with comprehensive testing and evaluation tools.

## Core Commands

### Environment Setup
```bash
# Create virtual environment (Python 3.11 required)
pipenv --python 3.11

# Install dependencies
pipenv run pip install mlx mlx-lm datasets pandas transformers huggingface-hub
```

### 1. Dataset Preparation
```bash
# SQL dataset (text-to-SQL, 106K examples)
pipenv run python dataset.py --type sql --sample-size 1000

# Alpaca dataset (instruction following, 52K examples)
pipenv run python dataset.py --type alpaca --sample-size 1000

# Full datasets (no sampling)
pipenv run python dataset.py --type sql
pipenv run python dataset.py --type alpaca
```

### 2. Training
```bash
# Quick test training (for development/testing)
pipenv run python train.py --dataset-type sql --quick-test
pipenv run python train.py --dataset-type alpaca --quick-test

# Standard training
pipenv run python train.py --dataset-type sql
pipenv run python train.py --dataset-type alpaca

# Conservative training (anti-overfitting)
pipenv run python train.py --dataset-type sql --conservative
pipenv run python train.py --dataset-type alpaca --conservative
```

### 3. Testing
```bash
# Interactive chat with fine-tuned model only
pipenv run python test_finetuned.py --adapter-path checkpoints --interactive

# Interactive chat with BOTH models (recommended!)
pipenv run python test_finetuned.py --adapter-path checkpoints --interactive --show-base

# Automated comparison with validation data
pipenv run python test_finetuned.py --adapter-path checkpoints --use-validation --num-samples 5

# Quick test with default prompts
pipenv run python test_finetuned.py --adapter-path checkpoints --quick-test
```

### 4. Benchmarking
```bash
# Performance benchmarking (uses validation data by default)
pipenv run python benchmark.py --adapter-path checkpoints --num-samples 10

# Compare base vs fine-tuned performance
pipenv run python benchmark.py --adapter-path checkpoints --compare --num-samples 10

# Quick benchmark with fewer samples
pipenv run python benchmark.py --adapter-path checkpoints --quick --num-samples 5
```

## Architecture

### 4-File Core Pipeline
- **`dataset.py`**: Dataset loading, formatting with Gemma chat templates, train/validation splitting
- **`train.py`**: LoRA fine-tuning using MLX, configuration management, training execution
- **`test_finetuned.py`**: Side-by-side model comparison, quality evaluation, response analysis
- **`benchmark.py`**: Performance benchmarking, memory profiling, speed analysis

### Key Classes and Configuration
- **`TrainingConfig`**: Centralized training parameters (LoRA settings, hyperparameters, paths)
- **`EvaluationConfig`**: Evaluation settings (generation parameters, test prompts)
- **`BenchmarkConfig`**: Benchmarking configuration (runs, memory profiling, metrics)

### Dataset Support
- **SQL Dataset** (`gretelai/synthetic_text_to_sql`): 106K text-to-SQL examples across 100+ domains (default)
- **Alpaca Dataset** (`yahma/alpaca-cleaned`): 52K instruction-following examples (default)
- **Custom Datasets**: Any HuggingFace dataset matching the required format

**Custom Dataset Usage:**
```bash
# Custom SQL dataset
pipenv run python dataset.py --type sql --dataset-name Clinton/Text-to-sql-v1

# Custom instruction-following dataset
pipenv run python dataset.py --type alpaca --dataset-name microsoft/orca-math-word-problems-200k
```

**Dataset Format Requirements:**
- **SQL datasets**: `sql_prompt`, `sql`, `sql_context` (optional), `sql_explanation` (optional)
- **Alpaca datasets**: `instruction`, `output`, `input` (optional)

### Training Modes
- **Quick Test**: 50-100 iterations, batch size 2, for development
- **Standard**: 1500-2000 iterations, batch size 4, regular training
- **Conservative**: 600-800 iterations, lower learning rate (5e-5), anti-overfitting

## Environment Requirements

- **macOS with Apple Silicon** (M-series chips)
- **Python 3.11**
- **HuggingFace Token**: Required for accessing Gemma 3 270M model
- **MLX Framework**: Apple Silicon optimized ML framework

### Required Environment Variables (.env)
```bash
HUGGINGFACE_HUB_TOKEN=hf_your_token_here
HF_HUB_ENABLE_HF_TRANSFER=1
TOKENIZERS_PARALLELISM=false
```

## Key Files and Directories

- `data/`: Generated training data (train.jsonl, valid.jsonl)
- `checkpoints/`: Model adapters and training outputs
  - `adapters.safetensors`: Latest trained model
  - `*_adapters.safetensors`: Checkpoint history
  - `training_config.txt`: Training configuration record
- `Pipfile`: Python dependencies with pinned versions
- `requirements.txt`: Alternative dependency specification

## Gemma Chat Template Format

The project uses Gemma's specific chat template:
```
<bos><start_of_turn>user
{prompt}<end_of_turn>
<start_of_turn>model
{response}<end_of_turn><eos>
```

This format is automatically applied during dataset preparation and inference.

## Model Access

- Requires access request at https://huggingface.co/google/gemma-3-270m-it
- Token must have read access to the gated model
- Verify access with: `pipenv run python -c "from huggingface_hub import whoami; print(whoami())"`

## Common Development Workflows

### Quick Development Test
```bash
pipenv run python dataset.py --type sql --sample-size 100
pipenv run python train.py --dataset-type sql --quick-test
pipenv run python test_finetuned.py --adapter-path checkpoints
```

### Production Training
```bash
pipenv run python dataset.py --type sql
pipenv run python train.py --dataset-type sql
pipenv run python benchmark.py --adapter-path checkpoints --compare
```