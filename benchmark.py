#!/usr/bin/env python3
"""
Performance benchmarking for Gemma models.

This script provides:
- Speed benchmarking (tokens/second)
- Memory usage profiling
- Latency measurements
- Throughput analysis
- Comparison between base and fine-tuned models
"""

import argparse
import time
import gc
import statistics
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import mlx.core as mx
from mlx_lm import load, generate

from dataset import BenchmarkConfig


def format_gemma_prompt(prompt: str) -> str:
    """Format a prompt using Gemma's chat template."""
    return f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    return mx.metal.get_cache_memory() / (1024**3)


def load_validation_samples(data_dir: str = "data", num_samples: int = 10) -> List[str]:
    """
    Load validation samples from JSONL file and extract user prompts.
    
    Args:
        data_dir: Directory containing validation data
        num_samples: Number of samples to extract for benchmarking
        
    Returns:
        List of user prompts extracted from validation data
    """
    validation_file = Path(data_dir) / "valid.jsonl"
    
    if not validation_file.exists():
        print(f"Warning: Validation file not found at {validation_file}")
        return []
    
    try:
        validation_texts = []
        with open(validation_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                validation_texts.append(data['text'])
        
        # Extract user prompts from Gemma chat templates
        user_prompts = []
        for text in validation_texts:
            # Find text between <start_of_turn>user and <end_of_turn>
            match = re.search(r'<start_of_turn>user\\n(.*?)<end_of_turn>', text, re.DOTALL)
            if match:
                prompt = match.group(1).strip()
                # Clean up escaped newlines
                prompt = prompt.replace('\\n\\n', '\n').replace('\\n', '\n')
                user_prompts.append(prompt)
        
        # Sample evenly from the validation set
        if len(user_prompts) > num_samples:
            step = len(user_prompts) // num_samples
            sampled_prompts = [user_prompts[i] for i in range(0, len(user_prompts), step)][:num_samples]
        else:
            sampled_prompts = user_prompts
        
        print(f"Loaded {len(sampled_prompts)} validation samples from {len(user_prompts)} total prompts")
        return sampled_prompts
        
    except Exception as e:
        print(f"Error loading validation data: {e}")
        return []


def benchmark_generation_speed(model, tokenizer, prompts: List[str], 
                             max_tokens: int = 128, warmup_runs: int = 3,
                             benchmark_runs: int = 10) -> Dict[str, Any]:
    """Benchmark generation speed for a model."""
    print(f"Running speed benchmark with {len(prompts)} prompts...")
    
    # Warmup runs
    print(f"Warming up with {warmup_runs} runs...")
    for i in range(warmup_runs):
        prompt = prompts[i % len(prompts)]
        formatted_prompt = format_gemma_prompt(prompt)
        _ = generate(model, tokenizer, formatted_prompt, max_tokens=max_tokens, verbose=False)
        gc.collect()
    
    # Benchmark runs
    print(f"Running {benchmark_runs} benchmark iterations...")
    all_times = []
    all_token_counts = []
    all_latencies = []
    
    for i in range(benchmark_runs):
        prompt = prompts[i % len(prompts)]
        formatted_prompt = format_gemma_prompt(prompt)
        
        # Measure first token latency
        start_time = time.time()
        
        # Generate response
        response = generate(model, tokenizer, formatted_prompt, max_tokens=max_tokens, verbose=False)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Count tokens in response
        response_tokens = len(tokenizer.encode(response))
        
        # Calculate metrics
        tokens_per_second = response_tokens / total_time if total_time > 0 else 0
        
        all_times.append(total_time)
        all_token_counts.append(response_tokens)
        all_latencies.append(total_time)
        
        print(f"  Run {i+1}: {response_tokens} tokens in {total_time:.2f}s ({tokens_per_second:.1f} tok/s)")
        
        gc.collect()
    
    # Calculate statistics
    speeds = [tokens / time for tokens, time in zip(all_token_counts, all_times) if time > 0]
    
    results = {
        "runs": benchmark_runs,
        "avg_tokens_per_second": statistics.mean(speeds) if speeds else 0,
        "median_tokens_per_second": statistics.median(speeds) if speeds else 0,
        "min_tokens_per_second": min(speeds) if speeds else 0,
        "max_tokens_per_second": max(speeds) if speeds else 0,
        "std_tokens_per_second": statistics.stdev(speeds) if len(speeds) > 1 else 0,
        "avg_latency": statistics.mean(all_latencies),
        "median_latency": statistics.median(all_latencies),
        "avg_tokens_generated": statistics.mean(all_token_counts),
        "total_tokens": sum(all_token_counts),
        "total_time": sum(all_times)
    }
    
    return results


def benchmark_memory_usage(model, tokenizer, prompts: List[str],
                          max_tokens: int = 128) -> Dict[str, Any]:
    """Benchmark memory usage during generation."""
    print("Benchmarking memory usage...")
    
    # Baseline memory
    gc.collect()
    baseline_memory = get_memory_usage()
    
    memory_measurements = []
    
    for i, prompt in enumerate(prompts[:5]):  # Test with first 5 prompts
        formatted_prompt = format_gemma_prompt(prompt)
        
        # Memory before generation
        pre_memory = get_memory_usage()
        
        # Generate
        _ = generate(model, tokenizer, formatted_prompt, max_tokens=max_tokens, verbose=False)
        
        # Memory after generation
        post_memory = get_memory_usage()
        
        memory_measurements.append({
            "prompt_idx": i,
            "pre_generation": pre_memory,
            "post_generation": post_memory,
            "generation_delta": post_memory - pre_memory
        })
        
        print(f"  Prompt {i+1}: {pre_memory:.2f}GB -> {post_memory:.2f}GB (Δ{post_memory - pre_memory:+.2f}GB)")
    
    # Final cleanup and measurement
    gc.collect()
    final_memory = get_memory_usage()
    
    results = {
        "baseline_memory_gb": baseline_memory,
        "peak_memory_gb": max(m["post_generation"] for m in memory_measurements),
        "final_memory_gb": final_memory,
        "max_generation_delta_gb": max(m["generation_delta"] for m in memory_measurements),
        "avg_generation_delta_gb": statistics.mean(m["generation_delta"] for m in memory_measurements),
        "measurements": memory_measurements
    }
    
    return results


def benchmark_different_lengths(model, tokenizer, base_prompt: str,
                               token_limits: List[int] = [32, 64, 128, 256]) -> Dict[str, Any]:
    """Benchmark performance with different output lengths."""
    print(f"Benchmarking different output lengths: {token_limits}")
    
    results = {}
    formatted_prompt = format_gemma_prompt(base_prompt)
    
    for max_tokens in token_limits:
        print(f"  Testing {max_tokens} max tokens...")
        
        times = []
        token_counts = []
        
        # Run multiple iterations for each length
        for _ in range(5):
            start_time = time.time()
            response = generate(model, tokenizer, formatted_prompt, max_tokens=max_tokens, verbose=False)
            end_time = time.time()
            
            total_time = end_time - start_time
            response_tokens = len(tokenizer.encode(response))
            
            times.append(total_time)
            token_counts.append(response_tokens)
        
        # Calculate average performance
        avg_time = statistics.mean(times)
        avg_tokens = statistics.mean(token_counts)
        avg_speed = avg_tokens / avg_time if avg_time > 0 else 0
        
        results[max_tokens] = {
            "avg_time": avg_time,
            "avg_tokens_generated": avg_tokens,
            "avg_tokens_per_second": avg_speed,
            "utilization": avg_tokens / max_tokens  # How much of max_tokens was used
        }
        
        print(f"    {avg_tokens:.0f} tokens in {avg_time:.2f}s ({avg_speed:.1f} tok/s)")
    
    return results


def run_comprehensive_benchmark(model, tokenizer, model_name: str,
                              config: BenchmarkConfig, data_dir: str = "data",
                              num_samples: int = 10) -> Dict[str, Any]:
    """Run comprehensive benchmark suite."""
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {model_name}")
    print(f"{'='*60}")
    
    # Load validation samples or use default test prompts
    validation_prompts = load_validation_samples(data_dir, num_samples)
    
    if validation_prompts:
        test_prompts = validation_prompts
        print(f"Using {len(test_prompts)} validation samples for benchmarking")
    else:
        # Fallback to default test prompts
        test_prompts = [
            "Explain the concept of machine learning in simple terms.",
            "Write a short story about a robot learning to paint.",
            "List the benefits of regular exercise.",
            "How do solar panels work?",
            "What are the primary colors and how do they mix?"
        ]
        print(f"Using {len(test_prompts)} default test prompts for benchmarking")
    
    results = {
        "model_name": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 1. Speed benchmark
    print("\n1. Speed Benchmark")
    print("-" * 30)
    speed_results = benchmark_generation_speed(
        model, tokenizer, test_prompts,
        max_tokens=config.max_tokens_benchmark,
        warmup_runs=config.warmup_runs,
        benchmark_runs=config.benchmark_runs
    )
    results["speed"] = speed_results
    
    # 2. Memory benchmark
    if config.profile_memory:
        print("\n2. Memory Usage Benchmark")
        print("-" * 30)
        memory_results = benchmark_memory_usage(
            model, tokenizer, test_prompts,
            max_tokens=config.max_tokens_benchmark
        )
        results["memory"] = memory_results
    
    # 3. Different length benchmark
    print("\n3. Variable Length Benchmark")
    print("-" * 30)
    length_results = benchmark_different_lengths(
        model, tokenizer, test_prompts[0]
    )
    results["variable_length"] = length_results
    
    return results


def compare_models(base_results: Dict[str, Any], 
                  finetuned_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare benchmark results between base and fine-tuned models."""
    comparison = {}
    
    # Speed comparison
    base_speed = base_results["speed"]["avg_tokens_per_second"]
    ft_speed = finetuned_results["speed"]["avg_tokens_per_second"]
    
    speed_change = (ft_speed - base_speed) / base_speed if base_speed > 0 else 0
    
    comparison["speed"] = {
        "base_tokens_per_second": base_speed,
        "finetuned_tokens_per_second": ft_speed,
        "speed_change_percent": speed_change * 100,
        "faster": ft_speed > base_speed
    }
    
    # Memory comparison
    if "memory" in base_results and "memory" in finetuned_results:
        base_memory = base_results["memory"]["peak_memory_gb"]
        ft_memory = finetuned_results["memory"]["peak_memory_gb"]
        
        memory_change = (ft_memory - base_memory) / base_memory if base_memory > 0 else 0
        
        comparison["memory"] = {
            "base_peak_memory_gb": base_memory,
            "finetuned_peak_memory_gb": ft_memory,
            "memory_change_percent": memory_change * 100,
            "more_efficient": ft_memory < base_memory
        }
    
    return comparison


def print_benchmark_summary(results: Dict[str, Any]):
    """Print a formatted summary of benchmark results."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY: {results['model_name']}")
    print(f"{'='*60}")
    
    # Speed metrics
    speed = results["speed"]
    print(f"SPEED PERFORMANCE:")
    print(f"  Average: {speed['avg_tokens_per_second']:.1f} tokens/second")
    print(f"  Median:  {speed['median_tokens_per_second']:.1f} tokens/second")
    print(f"  Range:   {speed['min_tokens_per_second']:.1f} - {speed['max_tokens_per_second']:.1f} tokens/second")
    print(f"  Std Dev: {speed['std_tokens_per_second']:.1f} tokens/second")
    print(f"  Avg Latency: {speed['avg_latency']:.2f} seconds")
    
    # Memory metrics
    if "memory" in results:
        memory = results["memory"]
        print(f"\nMEMORY USAGE:")
        print(f"  Peak Memory: {memory['peak_memory_gb']:.2f} GB")
        print(f"  Baseline:    {memory['baseline_memory_gb']:.2f} GB")
        print(f"  Max Delta:   +{memory['max_generation_delta_gb']:.2f} GB per generation")
    
    # Variable length performance
    print(f"\nVARIABLE LENGTH PERFORMANCE:")
    for max_tokens, metrics in results["variable_length"].items():
        utilization = metrics["utilization"] * 100
        print(f"  {max_tokens:3d} tokens: {metrics['avg_tokens_per_second']:5.1f} tok/s "
              f"({utilization:4.1f}% utilization)")


def print_comparison_summary(comparison: Dict[str, Any]):
    """Print comparison summary between models."""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    # Speed comparison
    speed = comparison["speed"]
    print(f"SPEED COMPARISON:")
    print(f"  Base model:       {speed['base_tokens_per_second']:.1f} tokens/second")
    print(f"  Fine-tuned model: {speed['finetuned_tokens_per_second']:.1f} tokens/second")
    print(f"  Change:           {speed['speed_change_percent']:+.1f}%")
    
    if speed["faster"]:
        print("  ✓ Fine-tuned model is faster")
    else:
        print("  ⚠ Fine-tuned model is slower")
    
    # Memory comparison
    if "memory" in comparison:
        memory = comparison["memory"]
        print(f"\nMEMORY COMPARISON:")
        print(f"  Base model:       {memory['base_peak_memory_gb']:.2f} GB")
        print(f"  Fine-tuned model: {memory['finetuned_peak_memory_gb']:.2f} GB")
        print(f"  Change:           {memory['memory_change_percent']:+.1f}%")
        
        if memory["more_efficient"]:
            print("  ✓ Fine-tuned model uses less memory")
        else:
            print("  ⚠ Fine-tuned model uses more memory")


def save_benchmark_results(results: Dict[str, Any], filename: str):
    """Save benchmark results to a JSON file."""
    import json
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nBenchmark results saved to: {filename}")


def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(description="Benchmark Gemma model performance")
    
    parser.add_argument("--model", type=str, default="google/gemma-3-270m-it",
                       help="Base model name")
    parser.add_argument("--adapter-path", type=str,
                       help="Path to fine-tuned adapter (optional)")
    parser.add_argument("--output-file", type=str, default="benchmark_results.json",
                       help="File to save benchmark results")
    parser.add_argument("--benchmark-runs", type=int, default=10,
                       help="Number of benchmark runs")
    parser.add_argument("--max-tokens", type=int, default=128,
                       help="Maximum tokens for benchmark")
    parser.add_argument("--compare", action="store_true",
                       help="Compare base and fine-tuned models")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmark with fewer iterations")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Directory containing validation data")
    parser.add_argument("--num-samples", type=int, default=10,
                       help="Number of validation samples to use for benchmarking")
    
    args = parser.parse_args()
    
    # Create benchmark config
    config = BenchmarkConfig()
    if args.quick:
        config.benchmark_runs = 3
        config.warmup_runs = 1
    else:
        config.benchmark_runs = args.benchmark_runs
    
    config.max_tokens_benchmark = args.max_tokens
    
    print("Starting benchmark suite...")
    print(f"Config: {config.benchmark_runs} runs, {config.max_tokens_benchmark} max tokens")
    
    if args.compare and args.adapter_path:
        # Compare base vs fine-tuned
        print("\nRunning comparison benchmark...")
        
        # Benchmark base model
        print("Loading base model...")
        base_model, base_tokenizer = load(args.model)
        base_results = run_comprehensive_benchmark(base_model, base_tokenizer, "Base Model", config, args.data_dir, args.num_samples)
        print_benchmark_summary(base_results)
        
        # Benchmark fine-tuned model
        print(f"\nLoading fine-tuned model with adapters: {args.adapter_path}")
        ft_model, ft_tokenizer = load(args.model, adapter_path=args.adapter_path)
        ft_results = run_comprehensive_benchmark(ft_model, ft_tokenizer, "Fine-tuned Model", config, args.data_dir, args.num_samples)
        print_benchmark_summary(ft_results)
        
        # Compare results
        comparison = compare_models(base_results, ft_results)
        print_comparison_summary(comparison)
        
        # Save combined results
        combined_results = {
            "base_model": base_results,
            "finetuned_model": ft_results,
            "comparison": comparison
        }
        save_benchmark_results(combined_results, args.output_file)
        
    else:
        # Benchmark single model
        if args.adapter_path:
            print(f"Loading fine-tuned model with adapters: {args.adapter_path}")
            model, tokenizer = load(args.model, adapter_path=args.adapter_path)
            model_name = "Fine-tuned Model"
        else:
            print("Loading base model...")
            model, tokenizer = load(args.model)
            model_name = "Base Model"
        
        results = run_comprehensive_benchmark(model, tokenizer, model_name, config, args.data_dir, args.num_samples)
        print_benchmark_summary(results)
        save_benchmark_results(results, args.output_file)
    
    print("\nBenchmarking completed!")


if __name__ == "__main__":
    main()