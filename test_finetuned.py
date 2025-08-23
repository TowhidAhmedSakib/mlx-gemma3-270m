#!/usr/bin/env python3
"""
Comprehensive testing suite for fine-tuned Gemma models.

This script provides:
- Side-by-side comparison of base vs fine-tuned model
- Evaluation on diverse test prompts
- Quality assessment and metrics
- Detailed output analysis
"""

import argparse
import time
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import mlx.core as mx
from mlx_lm import load, generate

from dataset import EvaluationConfig, create_evaluation_config


def print_evaluation_config(config: EvaluationConfig) -> None:
    """Print evaluation configuration in a readable format."""
    print("Evaluation Configuration:")
    print("=" * 40)
    print(f"Model: {config.model_name}")
    if config.adapter_path:
        print(f"Adapter Path: {config.adapter_path}")
    print(f"Max Tokens: {config.max_tokens}")
    print(f"Temperature: {config.temperature}")
    print(f"Number of Test Prompts: {len(config.test_prompts)}")
    print("=" * 40)


def format_gemma_prompt(prompt: str) -> str:
    """Format a prompt using Gemma's chat template for inference."""
    return f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


def load_validation_samples(data_dir: str = "data", num_samples: int = 5) -> List[str]:
    """
    Load validation samples from JSONL file and extract user prompts.
    
    Args:
        data_dir: Directory containing validation data
        num_samples: Number of samples to extract for testing
        
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
        
        # Sample evenly from the validation set for diverse testing
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


def generate_response(model, tokenizer, prompt: str, config: EvaluationConfig) -> Dict[str, Any]:
    """Generate a response and collect metrics."""
    formatted_prompt = format_gemma_prompt(prompt)
    
    start_time = time.time()
    
    try:
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            max_tokens=config.max_tokens,
            verbose=False
        )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Count tokens (approximate)
        response_tokens = len(tokenizer.encode(response))
        tokens_per_second = response_tokens / generation_time if generation_time > 0 else 0
        
        return {
            "response": response,
            "generation_time": generation_time,
            "response_tokens": response_tokens,
            "tokens_per_second": tokens_per_second,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        return {
            "response": f"Error: {str(e)}",
            "generation_time": 0,
            "response_tokens": 0,
            "tokens_per_second": 0,
            "success": False,
            "error": str(e)
        }


def evaluate_quality(prompt: str, response: str) -> Dict[str, float]:
    """Simple quality evaluation metrics."""
    metrics = {}
    
    # Length-based metrics
    metrics["response_length"] = len(response)
    metrics["word_count"] = len(response.split())
    
    # Basic coherence checks
    metrics["has_content"] = 1.0 if len(response.strip()) > 10 else 0.0
    metrics["not_repetitive"] = 1.0 if not _is_repetitive(response) else 0.0
    metrics["appropriate_length"] = 1.0 if 20 <= len(response.split()) <= 200 else 0.0
    
    # Instruction following (basic heuristics)
    if "write" in prompt.lower() and "haiku" in prompt.lower():
        metrics["follows_format"] = 1.0 if _check_haiku_format(response) else 0.0
    elif "list" in prompt.lower():
        metrics["follows_format"] = 1.0 if _check_list_format(response) else 0.0
    else:
        metrics["follows_format"] = 0.5  # Default neutral score
    
    return metrics


def _is_repetitive(text: str, threshold: float = 0.7) -> bool:
    """Check if text is overly repetitive."""
    words = text.split()
    if len(words) < 10:
        return False
    
    # Check for repeated phrases
    unique_words = set(words)
    repetition_ratio = len(unique_words) / len(words)
    
    return repetition_ratio < threshold


def _check_haiku_format(text: str) -> bool:
    """Basic check for haiku-like structure."""
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    return len(lines) == 3


def _check_list_format(text: str) -> bool:
    """Basic check for list structure."""
    list_indicators = ['1.', '2.', '3.', '-', '*', '•']
    return any(indicator in text for indicator in list_indicators)


def run_comparison_test(base_model, base_tokenizer, 
                       finetuned_model, finetuned_tokenizer,
                       prompts: List[str], config: EvaluationConfig) -> List[Dict]:
    """Run side-by-side comparison tests."""
    results = []
    
    print(f"Running comparison tests on {len(prompts)} prompts...")
    print("=" * 80)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nTest {i}/{len(prompts)}: {prompt}")
        print("-" * 60)
        
        # Generate with base model
        print("Base model response:")
        base_result = generate_response(base_model, base_tokenizer, prompt, config)
        print(base_result["response"])
        
        print("\nFine-tuned model response:")
        # Generate with fine-tuned model
        ft_result = generate_response(finetuned_model, finetuned_tokenizer, prompt, config)
        print(ft_result["response"])
        
        # Evaluate quality
        base_quality = evaluate_quality(prompt, base_result["response"])
        ft_quality = evaluate_quality(prompt, ft_result["response"])
        
        # Store results
        test_result = {
            "prompt": prompt,
            "base_response": base_result["response"],
            "finetuned_response": ft_result["response"],
            "base_metrics": {**base_result, **base_quality},
            "finetuned_metrics": {**ft_result, **ft_quality},
        }
        results.append(test_result)
        
        print(f"\nMetrics comparison:")
        print(f"Base - Tokens/sec: {base_result['tokens_per_second']:.1f}, Quality: {base_quality['not_repetitive']:.1f}")
        print(f"Fine-tuned - Tokens/sec: {ft_result['tokens_per_second']:.1f}, Quality: {ft_quality['not_repetitive']:.1f}")
        print("=" * 80)
    
    return results


def generate_summary(results: List[Dict]) -> Dict[str, Any]:
    """Generate summary statistics from test results."""
    summary = {
        "total_tests": len(results),
        "base_model": {},
        "finetuned_model": {},
        "improvements": {}
    }
    
    # Aggregate metrics
    base_speeds = [r["base_metrics"]["tokens_per_second"] for r in results if r["base_metrics"]["success"]]
    ft_speeds = [r["finetuned_metrics"]["tokens_per_second"] for r in results if r["finetuned_metrics"]["success"]]
    
    base_quality = [r["base_metrics"]["not_repetitive"] for r in results]
    ft_quality = [r["finetuned_metrics"]["not_repetitive"] for r in results]
    
    # Base model stats
    summary["base_model"] = {
        "avg_tokens_per_second": sum(base_speeds) / len(base_speeds) if base_speeds else 0,
        "avg_quality_score": sum(base_quality) / len(base_quality) if base_quality else 0,
        "success_rate": sum(1 for r in results if r["base_metrics"]["success"]) / len(results)
    }
    
    # Fine-tuned model stats
    summary["finetuned_model"] = {
        "avg_tokens_per_second": sum(ft_speeds) / len(ft_speeds) if ft_speeds else 0,
        "avg_quality_score": sum(ft_quality) / len(ft_quality) if ft_quality else 0,
        "success_rate": sum(1 for r in results if r["finetuned_metrics"]["success"]) / len(results)
    }
    
    # Improvements
    if summary["base_model"]["avg_quality_score"] > 0:
        quality_improvement = (summary["finetuned_model"]["avg_quality_score"] - 
                             summary["base_model"]["avg_quality_score"]) / summary["base_model"]["avg_quality_score"]
        summary["improvements"]["quality_improvement"] = quality_improvement
    
    return summary


def print_summary(summary: Dict[str, Any]):
    """Print a formatted summary of test results."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    print(f"Total tests conducted: {summary['total_tests']}")
    print()
    
    print("BASE MODEL PERFORMANCE:")
    print(f"  Average tokens/second: {summary['base_model']['avg_tokens_per_second']:.1f}")
    print(f"  Average quality score: {summary['base_model']['avg_quality_score']:.2f}")
    print(f"  Success rate: {summary['base_model']['success_rate']:.1%}")
    print()
    
    print("FINE-TUNED MODEL PERFORMANCE:")
    print(f"  Average tokens/second: {summary['finetuned_model']['avg_tokens_per_second']:.1f}")
    print(f"  Average quality score: {summary['finetuned_model']['avg_quality_score']:.2f}")
    print(f"  Success rate: {summary['finetuned_model']['success_rate']:.1%}")
    print()
    
    if "quality_improvement" in summary["improvements"]:
        improvement = summary["improvements"]["quality_improvement"]
        print(f"QUALITY IMPROVEMENT: {improvement:+.1%}")
        if improvement > 0:
            print("✓ Fine-tuning improved response quality")
        else:
            print("⚠ Fine-tuning may have reduced response quality")
    
    print("=" * 80)


def interactive_chat(base_model, base_tokenizer, finetuned_model, finetuned_tokenizer,
                    config: EvaluationConfig, show_base: bool = False):
    """Run interactive chat session with the fine-tuned model."""
    print("\n" + "="*80)
    print("🤖 INTERACTIVE CHAT MODE")
    print("="*80)
    print("Chat with your fine-tuned model! Type your questions below.")
    print("Commands: 'quit', 'exit', 'bye' to stop")
    if show_base:
        print("💡 Showing both base and fine-tuned responses for comparison")
    print("="*80)
    
    try:
        while True:
            # Get user input
            user_input = input("\n💬 You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\n👋 Goodbye! Chat session ended.")
                break
            
            if not user_input:
                continue
            
            print("\n" + "-"*60)
            
            # Show base model response if requested
            if show_base:
                print("🔹 Base Model:")
                base_result = generate_response(base_model, base_tokenizer, user_input, config)
                if base_result["success"]:
                    print(base_result["response"])
                    print(f"⚡ {base_result['tokens_per_second']:.1f} tok/s, {base_result['generation_time']:.2f}s")
                else:
                    print(f"❌ Error: {base_result['error']}")
                print("\n" + "-"*30)
            
            # Show fine-tuned model response
            print("🔸 Fine-tuned Model:" if show_base else "🤖 Response:")
            ft_result = generate_response(finetuned_model, finetuned_tokenizer, user_input, config)
            if ft_result["success"]:
                print(ft_result["response"])
                print(f"⚡ {ft_result['tokens_per_second']:.1f} tok/s, {ft_result['generation_time']:.2f}s")
            else:
                print(f"❌ Error: {ft_result['error']}")
            
            print("-"*60)
            
    except KeyboardInterrupt:
        print("\n\n👋 Chat interrupted. Goodbye!")
    except EOFError:
        print("\n\n👋 Chat ended. Goodbye!")


def save_results(results: List[Dict], output_file: str):
    """Save detailed results to a file."""
    import json
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_file}")


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="Test fine-tuned Gemma model")
    
    parser.add_argument("--adapter-path", type=str, required=True,
                       help="Path to the fine-tuned adapter file")
    parser.add_argument("--model", type=str, default="google/gemma-3-270m-it",
                       help="Base model name")
    parser.add_argument("--max-tokens", type=int, default=256,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--output-file", type=str, default="test_results.json",
                       help="File to save detailed results")
    parser.add_argument("--custom-prompts", type=str, nargs="+",
                       help="Custom prompts to test")
    parser.add_argument("--quick-test", action="store_true",
                       help="Run only a subset of tests")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Directory containing validation data")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of validation samples to test")
    parser.add_argument("--use-validation", action="store_true",
                       help="Use validation data instead of default prompts")
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive chat mode")
    parser.add_argument("--show-base", action="store_true",
                       help="Show base model responses in interactive mode")
    
    args = parser.parse_args()
    
    # Create evaluation config
    config = create_evaluation_config(
        model_name=args.model,
        adapter_path=args.adapter_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    # Determine which prompts to use
    if args.custom_prompts:
        config.test_prompts = args.custom_prompts
        print(f"Using {len(config.test_prompts)} custom prompts")
    elif args.use_validation:
        validation_prompts = load_validation_samples(args.data_dir, args.num_samples)
        if validation_prompts:
            config.test_prompts = validation_prompts
            print(f"Using {len(config.test_prompts)} validation prompts for quality comparison")
        else:
            print("Failed to load validation data, using default prompts")
            if args.quick_test:
                config.test_prompts = config.test_prompts[:5]
    elif args.quick_test:
        config.test_prompts = config.test_prompts[:5]  # Use first 5 prompts
        print(f"Using {len(config.test_prompts)} default prompts (quick test mode)")
    
    print_evaluation_config(config)
    
    # Load models
    print("Loading base model...")
    base_model, base_tokenizer = load(config.model_name)
    
    print(f"Loading fine-tuned model with adapters: {config.adapter_path}")
    if not Path(config.adapter_path).exists():
        print(f"Error: Adapter file not found: {config.adapter_path}")
        return
    
    finetuned_model, finetuned_tokenizer = load(
        config.model_name, 
        adapter_path=config.adapter_path
    )
    
    # Handle interactive mode
    if args.interactive:
        interactive_chat(
            base_model, base_tokenizer,
            finetuned_model, finetuned_tokenizer,
            config, args.show_base
        )
        return
    
    # Run comparison tests
    results = run_comparison_test(
        base_model, base_tokenizer,
        finetuned_model, finetuned_tokenizer,
        config.test_prompts, config
    )
    
    # Generate and print summary
    summary = generate_summary(results)
    print_summary(summary)
    
    # Save detailed results
    save_results(results, args.output_file)
    
    print(f"\nTesting completed! Use --interactive for manual testing.")


if __name__ == "__main__":
    main()