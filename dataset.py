#!/usr/bin/env python3
"""
Dataset preparation for Gemma 3 270M fine-tuning.

Supports both Alpaca (general instruction following) and SQL (text-to-SQL) datasets
with proper Gemma chat template formatting.

Usage:
    python dataset.py --type sql --sample-size 1000
    python dataset.py --type alpaca --sample-size 5000
    python dataset.py --help
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from datasets import load_dataset


# Configuration classes for shared use
@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    
    # Model settings
    model_name: str = "google/gemma-3-270m-it"
    
    # Dataset settings
    dataset_type: str = "alpaca"  # "alpaca" or "sql"
    dataset_name: Optional[str] = None  # Auto-selected based on dataset_type if None
    sample_size: Optional[int] = None  # Use full dataset by default
    train_ratio: float = 0.9
    data_dir: str = "data"
    
    # LoRA settings
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Training hyperparameters
    learning_rate: float = 2e-4
    batch_size: int = 4
    max_iters: int = 1500
    steps_per_report: int = 50
    steps_per_eval: int = 250
    save_every: int = 250
    
    # Output settings
    output_dir: str = "checkpoints"
    adapter_filename: str = "adapters.npz"
    
    # Training options
    grad_checkpoint: bool = True
    seed: int = 42
    
    def get_adapter_path(self, iteration: Optional[int] = None) -> str:
        """Get the path to saved adapters."""
        if iteration:
            return f"{self.output_dir}/{iteration}_{self.adapter_filename}"
        else:
            return f"{self.output_dir}/{self.adapter_filename}"


@dataclass 
class EvaluationConfig:
    """Evaluation and testing configuration."""
    
    # Model settings
    model_name: str = "google/gemma-3-270m-it"
    adapter_path: Optional[str] = None
    
    # Generation settings
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    
    # Test prompts for evaluation
    test_prompts: list = None
    
    def __post_init__(self):
        if self.test_prompts is None:
            self.test_prompts = self._get_default_prompts()

    def _get_default_prompts(self):
        """Get default test prompts based on model type."""
        return [
            "Write a haiku about machine learning.",
            "Explain the concept of gravity in simple terms.",
            "List 5 benefits of regular exercise.",
            "What is the capital of France?",
            "How do solar panels work?",
            "Tell me a short story about a robot who learns to paint.",
            "If I have 10 apples and give away 3, then buy 5 more, how many do I have?",
            "Write a Python function to calculate the factorial of a number.",
        ]
    
    def get_sql_prompts(self):
        """Get SQL-specific test prompts for text-to-SQL evaluation."""
        return [
            "Show all customers from New York.\\n\\nContext: CREATE TABLE customers (id INT, name VARCHAR(100), city VARCHAR(50));",
            "Count the total number of orders.\\n\\nContext: CREATE TABLE orders (id INT, customer_id INT, order_date DATE);",
            "Find all products with price greater than 100.\\n\\nContext: CREATE TABLE products (id INT, name VARCHAR(100), price DECIMAL(10,2));",
            "Find the average salary by department.\\n\\nContext: CREATE TABLE employees (id INT, name VARCHAR(100), salary DECIMAL(10,2), department VARCHAR(50));",
            "Find the top 5 customers by total order value.\\n\\nContext: CREATE TABLE customers (id INT, name VARCHAR(100)); CREATE TABLE orders (id INT, customer_id INT, total DECIMAL(10,2));",
        ]


@dataclass
class BenchmarkConfig:
    """Benchmarking configuration."""
    
    # Performance testing
    warmup_runs: int = 3
    benchmark_runs: int = 10
    max_tokens_benchmark: int = 128
    
    # Memory profiling
    profile_memory: bool = True
    
    # Quality metrics
    calculate_bleu: bool = False  # Requires reference responses
    reference_responses: Optional[dict] = None


# Dataset formatting functions
def format_gemma_chat(instruction: str, response: str, input_text: str = "") -> str:
    """
    Format instruction and response using Gemma's chat template.
    
    Args:
        instruction: The instruction/prompt
        response: The expected response
        input_text: Optional additional input context
        
    Returns:
        Formatted string following Gemma chat template
    """
    if input_text.strip():
        user_content = f"{instruction}\\n\\nInput: {input_text}"
    else:
        user_content = instruction
        
    return (
        "<bos><start_of_turn>user\\n"
        f"{user_content}<end_of_turn>\\n"
        "<start_of_turn>model\\n"
        f"{response}<end_of_turn><eos>"
    )


def format_sql_instruction(sql_prompt: str, sql_context: str, sql_query: str, 
                          sql_explanation: str = "") -> str:
    """
    Format SQL instruction using Gemma's chat template.
    
    Args:
        sql_prompt: The natural language query/request
        sql_context: The database schema or table context
        sql_query: The SQL query response
        sql_explanation: Optional explanation of the SQL query
        
    Returns:
        Formatted string following Gemma chat template for SQL tasks
    """
    # Build user content with prompt and context
    if sql_context.strip():
        user_content = f"{sql_prompt}\\n\\nContext: {sql_context}"
    else:
        user_content = sql_prompt
    
    # Build model response with SQL and optional explanation
    if sql_explanation.strip():
        model_response = f"{sql_query}\\n\\nExplanation: {sql_explanation}"
    else:
        model_response = sql_query
        
    return (
        "<bos><start_of_turn>user\\n"
        f"{user_content}<end_of_turn>\\n"
        "<start_of_turn>model\\n"
        f"{model_response}<end_of_turn><eos>"
    )


# Dataset loading functions
def load_alpaca_dataset(dataset_name: str = "yahma/alpaca-cleaned", 
                       sample_size: Optional[int] = None,
                       seed: int = 42) -> pd.DataFrame:
    """Load and preprocess the Alpaca dataset."""
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    # Convert to DataFrame
    df = pd.DataFrame(dataset["train"])
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed)
        print(f"Sampled {sample_size} examples from {len(dataset['train'])} total")
    
    # Format using Gemma chat template
    print("Formatting examples with Gemma chat template...")
    df["text"] = df.apply(
        lambda row: format_gemma_chat(
            instruction=row["instruction"],
            response=row["output"],
            input_text=row.get("input", "")
        ),
        axis=1
    )
    
    print(f"Prepared {len(df)} training examples")
    return df


def load_sql_dataset(dataset_name: str = "gretelai/synthetic_text_to_sql",
                    sample_size: Optional[int] = None,
                    seed: int = 42) -> pd.DataFrame:
    """Load and preprocess the SQL dataset."""
    print(f"Loading SQL dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    # Convert to DataFrame (use train split)
    df = pd.DataFrame(dataset["train"])
    
    print(f"Loaded {len(df)} SQL examples")
    print(f"Domains covered: {df['domain'].nunique()}")
    print(f"SQL complexity levels: {df['sql_complexity'].unique()}")
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed)
        print(f"Sampled {sample_size} examples from {len(dataset['train'])} total")
    
    # Format using SQL-specific Gemma chat template
    print("Formatting SQL examples with Gemma chat template...")
    df["text"] = df.apply(
        lambda row: format_sql_instruction(
            sql_prompt=row["sql_prompt"],
            sql_context=row.get("sql_context", ""),
            sql_query=row["sql"],
            sql_explanation=row.get("sql_explanation", "")
        ),
        axis=1
    )
    
    print(f"Prepared {len(df)} SQL training examples")
    return df


def split_dataset(df: pd.DataFrame, 
                 train_ratio: float = 0.9,
                 seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into train and validation sets."""
    # Shuffle the dataset
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    # Calculate split point
    split_idx = int(len(df_shuffled) * train_ratio)
    
    train_df = df_shuffled[:split_idx]
    val_df = df_shuffled[split_idx:]
    
    print(f"Split dataset: {len(train_df)} train, {len(val_df)} validation")
    return train_df, val_df


def save_dataset_splits(train_df: pd.DataFrame, 
                       val_df: pd.DataFrame,
                       output_dir: str = "data") -> None:
    """Save train and validation splits to JSONL files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save as JSONL files (required by MLX)
    train_path = output_path / "train.jsonl"
    val_path = output_path / "valid.jsonl"
    
    train_df[["text"]].to_json(train_path, orient="records", lines=True, force_ascii=False)
    val_df[["text"]].to_json(val_path, orient="records", lines=True, force_ascii=False)
    
    print(f"Saved training data to {train_path}")
    print(f"Saved validation data to {val_path}")


def prepare_dataset(dataset_type: str = "alpaca",
                   dataset_name: Optional[str] = None,
                   sample_size: Optional[int] = None,
                   train_ratio: float = 0.9,
                   output_dir: str = "data",
                   seed: int = 42) -> None:
    """
    Complete pipeline to prepare dataset for training.
    
    Args:
        dataset_type: Type of dataset ("alpaca" or "sql")
        dataset_name: HuggingFace dataset identifier (optional, uses defaults)
        sample_size: Optional size to sample from dataset
        train_ratio: Fraction of data to use for training
        output_dir: Directory to save processed files
        seed: Random seed for reproducibility
    """
    # Set default dataset names if not provided
    if dataset_name is None:
        if dataset_type == "alpaca":
            dataset_name = "yahma/alpaca-cleaned"
        elif dataset_type == "sql":
            dataset_name = "gretelai/synthetic_text_to_sql"
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Load and format dataset based on type
    if dataset_type == "alpaca":
        df = load_alpaca_dataset(dataset_name, sample_size, seed)
    elif dataset_type == "sql":
        df = load_sql_dataset(dataset_name, sample_size, seed)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Use 'alpaca' or 'sql'")
    
    # Split into train/val
    train_df, val_df = split_dataset(df, train_ratio, seed)
    
    # Save to files
    save_dataset_splits(train_df, val_df, output_dir)
    
    # Print sample
    print(f"\\nSample {dataset_type} formatted example:")
    print("-" * 50)
    print(train_df.iloc[0]["text"])
    print("-" * 50)


# Configuration helper functions
def create_training_config(**kwargs) -> TrainingConfig:
    """Create a training config with custom parameters."""
    config = TrainingConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config parameter: {key}")
    return config


def create_evaluation_config(**kwargs) -> EvaluationConfig:
    """Create an evaluation config with custom parameters.""" 
    config = EvaluationConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config parameter: {key}")
    return config


# Command line interface
def main():
    """Command line interface for dataset preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare datasets for Gemma 3 270M fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dataset.py --type sql --sample-size 1000
  python dataset.py --type alpaca --sample-size 5000  
  python dataset.py --type sql --output-dir my_data
        """
    )
    
    parser.add_argument("--type", type=str, default="sql",
                       choices=["alpaca", "sql"],
                       help="Dataset type (default: sql)")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Sample size (uses full dataset if not specified)")
    parser.add_argument("--train-ratio", type=float, default=0.9,
                       help="Training split ratio (default: 0.9)")
    parser.add_argument("--output-dir", type=str, default="data",
                       help="Output directory (default: data)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--dataset-name", type=str, default=None,
                       help="Custom dataset name (overrides default)")
    
    args = parser.parse_args()
    
    print(f"Preparing {args.type} dataset...")
    if args.sample_size:
        print(f"Sample size: {args.sample_size:,}")
    else:
        print("Using full dataset")
    
    prepare_dataset(
        dataset_type=args.type,
        dataset_name=args.dataset_name,
        sample_size=args.sample_size,
        train_ratio=args.train_ratio,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    print("\\n✅ Dataset preparation completed!")
    print(f"📁 Files created in {args.output_dir}/:")
    print(f"   - train.jsonl")
    print(f"   - valid.jsonl")
    print("\\n🚀 Next step: Run training with:")
    print(f"   python train.py --dataset-type {args.type}")


if __name__ == "__main__":
    main()