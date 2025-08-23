#!/usr/bin/env python3
"""
Fine-tune Gemma 3 270M using LoRA with MLX.

This script provides a complete training pipeline with:
- Dataset loading and preprocessing
- LoRA fine-tuning using MLX
- Checkpointing and monitoring
- Validation evaluation
"""

import argparse
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from dataset import TrainingConfig, create_training_config, prepare_dataset


def print_training_config(config: TrainingConfig) -> None:
    """Print training configuration in a readable format."""
    print("Training Configuration:")
    print("=" * 40)
    print(f"Model: {config.model_name}")
    print(f"Dataset Type: {config.dataset_type}")
    if config.dataset_name:
        print(f"Dataset: {config.dataset_name}")
    else:
        default_datasets = {"alpaca": "yahma/alpaca-cleaned", "sql": "gretelai/synthetic_text_to_sql"}
        print(f"Dataset: {default_datasets.get(config.dataset_type, 'Unknown')}")
    if config.sample_size:
        print(f"Sample Size: {config.sample_size:,}")
    print(f"LoRA Rank: {config.lora_rank}, Alpha: {config.lora_alpha}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Max Iterations: {config.max_iters:,}")
    print(f"Output Directory: {config.output_dir}")
    print("=" * 40)


def setup_environment():
    """Set up environment variables for training."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def prepare_data(config: TrainingConfig):
    """Prepare training data according to config."""
    print(f"Preparing {config.dataset_type} training data...")
    
    # Check if data already exists
    data_dir = Path(config.data_dir)
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "valid.jsonl"
    
    if train_file.exists() and val_file.exists():
        print(f"Found existing data files in {config.data_dir}")
        return
    
    # Prepare fresh data using the new unified function
    prepare_dataset(
        dataset_type=config.dataset_type,
        dataset_name=config.dataset_name,
        sample_size=config.sample_size,
        train_ratio=config.train_ratio,
        output_dir=config.data_dir,
        seed=config.seed
    )


def run_training(config: TrainingConfig):
    """Execute the LoRA training using MLX."""
    print("Starting LoRA training...")
    
    # Ensure output directory exists
    Path(config.output_dir).mkdir(exist_ok=True)
    
    # Build MLX LoRA command - using new format
    cmd_parts = [
        "python", "-m", "mlx_lm", "lora",
        "--model", config.model_name,
        "--train",
        "--data", config.data_dir,
        "--iters", str(config.max_iters),
        "--batch-size", str(config.batch_size),
        "--learning-rate", str(config.learning_rate),
        "--steps-per-report", str(config.steps_per_report),
        "--steps-per-eval", str(config.steps_per_eval),
        "--num-layers", "16",  # Use 16 layers for 270M model
        "--adapter-path", config.output_dir,
        "--save-every", str(config.save_every),
    ]
    
    if config.grad_checkpoint:
        cmd_parts.append("--grad-checkpoint")
    
    print("Training command:")
    print(" ".join(cmd_parts))
    print()
    
    # Execute training
    import subprocess
    try:
        result = subprocess.run(cmd_parts, check=True, capture_output=False)
        print("Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        return False


def validate_training_setup(config: TrainingConfig):
    """Validate that training setup is correct."""
    print("Validating training setup...")
    
    # Check if model is accessible
    try:
        print(f"Loading model: {config.model_name}")
        model, tokenizer = load(config.model_name)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Check data directory
    data_dir = Path(config.data_dir)
    if not data_dir.exists():
        print(f"✓ Data directory will be created: {data_dir}")
    else:
        train_file = data_dir / "train.jsonl"
        val_file = data_dir / "valid.jsonl"
        if train_file.exists() and val_file.exists():
            print("✓ Training data files found")
        else:
            print("✓ Training data will be prepared")
    
    # Check output directory
    output_dir = Path(config.output_dir)
    if not output_dir.exists():
        print(f"✓ Output directory will be created: {output_dir}")
    else:
        print(f"✓ Output directory exists: {output_dir}")
    
    return True


def save_training_info(config: TrainingConfig):
    """Save training configuration and info for later reference."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save config as text file
    config_file = output_dir / "training_config.txt"
    with open(config_file, "w") as f:
        f.write("Training Configuration\n")
        f.write("=" * 40 + "\n")
        f.write(f"Model: {config.model_name}\n")
        f.write(f"Dataset: {config.dataset_name}\n")
        f.write(f"Sample Size: {config.sample_size}\n")
        f.write(f"LoRA Rank: {config.lora_rank}, Alpha: {config.lora_alpha}\n")
        f.write(f"Learning Rate: {config.learning_rate}\n")
        f.write(f"Batch Size: {config.batch_size}\n")
        f.write(f"Max Iterations: {config.max_iters}\n")
        f.write(f"Training started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Training info saved to {config_file}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 3 270M with LoRA")
    
    # Model and data arguments
    parser.add_argument("--model", type=str, default="google/gemma-3-270m-it",
                       help="Model name or path")
    parser.add_argument("--dataset-type", type=str, default="alpaca",
                       choices=["alpaca", "sql"],
                       help="Type of dataset to use (alpaca or sql)")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Dataset name from HuggingFace (auto-selected if not specified)")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Sample size from dataset (use full by default)")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Directory for processed data")
    
    # Training arguments
    parser.add_argument("--max-iters", type=int, default=1500,
                       help="Maximum training iterations")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                       help="LoRA alpha")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--save-every", type=int, default=250,
                       help="Save checkpoint every N iterations")
    
    # Training options
    parser.add_argument("--quick-test", action="store_true",
                       help="Run quick test with small dataset and few iterations")
    parser.add_argument("--conservative", action="store_true",
                       help="Use conservative anti-overfitting parameters")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip training setup validation")
    parser.add_argument("--prepare-data-only", action="store_true",
                       help="Only prepare data, don't run training")
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Create config from arguments
    if args.quick_test:
        print(f"Running in quick test mode with {args.dataset_type} dataset...")
        if args.dataset_type == "sql":
            # SQL Quick Test config
            config_dict = {
                'dataset_type': 'sql',
                'sample_size': 500,  # Larger test sample for SQL complexity
                'max_iters': 100,
                'batch_size': 2,
                'save_every': 50
            }
        else:
            config_dict = {
                'dataset_type': 'alpaca',
                'sample_size': 100,
                'max_iters': 50,
                'batch_size': 2,
                'save_every': 25
            }
        
        # Override with command line arguments
        config_dict.update({
            'model_name': args.model,
            'dataset_name': args.dataset,
            'dataset_type': args.dataset_type,
            'data_dir': args.data_dir,
            'output_dir': args.output_dir
        })
        
        # Override with any explicitly provided args
        if args.sample_size is not None:
            config_dict['sample_size'] = args.sample_size
            
        config = create_training_config(**config_dict)
    elif args.conservative:
        print(f"Running in conservative anti-overfitting mode with {args.dataset_type} dataset...")
        if args.dataset_type == "sql":
            # SQL Conservative config
            config_dict = {
                'dataset_type': 'sql',
                'learning_rate': 5e-5,  # Conservative learning rate
                'batch_size': 4,
                'max_iters': 800,  # Conservative iteration count
                'steps_per_eval': 100,
                'save_every': 100,
                'lora_dropout': 0.1  # Higher dropout for regularization
            }
        else:
            # Alpaca Conservative config
            config_dict = {
                'dataset_type': 'alpaca',
                'learning_rate': 5e-5,  # Reduced from 2e-4 to prevent overfitting
                'batch_size': 4,
                'max_iters': 600,  # Reduced from 1500 based on overfitting analysis
                'steps_per_eval': 100,  # More frequent validation checks
                'steps_per_report': 50,
                'save_every': 100,  # More frequent checkpoints for early stopping
                'lora_dropout': 0.05,  # Increased regularization
            }
        
        # Override with command line arguments
        config_dict.update({
            'model_name': args.model,
            'dataset_name': args.dataset,
            'dataset_type': args.dataset_type,
            'data_dir': args.data_dir,
            'output_dir': args.output_dir
        })
        
        # Override with any explicitly provided args
        if args.sample_size is not None:
            config_dict['sample_size'] = args.sample_size
        if args.max_iters != 1500:  # If user provided custom iters
            config_dict['max_iters'] = args.max_iters
        if args.batch_size != 4:  # If user provided custom batch size
            config_dict['batch_size'] = args.batch_size
        if args.learning_rate != 2e-4:  # If user provided custom LR
            config_dict['learning_rate'] = args.learning_rate
            
        config = create_training_config(**config_dict)
    else:
        print(f"Running standard training with {args.dataset_type} dataset...")
        if args.dataset_type == "sql":
            # SQL Training config with optimized parameters for code generation
            config_dict = {
                'dataset_type': 'sql',
                'learning_rate': 1e-4,  # Slightly lower for code generation
                'batch_size': 4,
                'max_iters': 2000,  # More iterations for larger dataset
                'steps_per_eval': 200,
                'save_every': 400,
                'lora_rank': 16,
                'lora_alpha': 32
            }
            # Override defaults with command line arguments
            config_dict.update({
                'model_name': args.model,
                'dataset_name': args.dataset,
                'dataset_type': args.dataset_type,
                'data_dir': args.data_dir,
                'output_dir': args.output_dir
            })
            # Override with explicitly provided arguments
            if args.sample_size is not None:
                config_dict['sample_size'] = args.sample_size
            if args.max_iters != 1500:
                config_dict['max_iters'] = args.max_iters
            if args.batch_size != 4:
                config_dict['batch_size'] = args.batch_size
            if args.learning_rate != 2e-4:
                config_dict['learning_rate'] = args.learning_rate
            if args.lora_rank != 16:
                config_dict['lora_rank'] = args.lora_rank
            if args.lora_alpha != 32:
                config_dict['lora_alpha'] = args.lora_alpha
            if args.save_every != 250:
                config_dict['save_every'] = args.save_every
            
            config = create_training_config(**config_dict)
        else:
            config = create_training_config(
                model_name=args.model,
                dataset_type=args.dataset_type,
                dataset_name=args.dataset,
                sample_size=args.sample_size,
                data_dir=args.data_dir,
                max_iters=args.max_iters,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                output_dir=args.output_dir,
                save_every=args.save_every
            )
    
    # Print configuration
    print_training_config(config)
    
    # Validate setup
    if not args.skip_validation:
        if not validate_training_setup(config):
            print("Setup validation failed. Use --skip-validation to override.")
            sys.exit(1)
    
    # Prepare data
    prepare_data(config)
    
    if args.prepare_data_only:
        print("Data preparation complete. Exiting.")
        return
    
    # Save training info
    save_training_info(config)
    
    # Run training
    print("\nStarting training...")
    success = run_training(config)
    
    if success:
        final_adapter_path = config.get_adapter_path(config.max_iters)
        print(f"\nTraining completed successfully!")
        print(f"Final adapters saved to: {final_adapter_path}")
        print(f"\nTo test your model, run:")
        print(f"python test_finetuned.py --adapter-path {final_adapter_path}")
    else:
        print("\nTraining failed. Check the logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()