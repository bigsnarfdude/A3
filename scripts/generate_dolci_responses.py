#!/usr/bin/env python3
"""
Generate model-specific responses for DOLCI dataset via OpenRouter.

This script loads the DOLCI-Instruct-SFT dataset, sends each prompt to a specified model
via OpenRouter using concurrent processing, and saves the responses to a JSON file.
The saved responses can then be used for mixing in iterative SFT training to prevent
catastrophic forgetting while maintaining the target model's response style.

Usage:
    # Generate Qwen responses
    python scripts/generate_dolci_responses.py --output data/dolci_qwen_responses.json \
        --model qwen/qwen-2.5-72b-instruct

    # Generate Llama responses
    python scripts/generate_dolci_responses.py --output data/dolci_llama_responses.json \
        --model meta-llama/llama-3.1-70b-instruct

Requirements:
    - OPENROUTER_API_KEY environment variable must be set
    - pip install datasets requests tqdm
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

try:
    import requests
except ImportError:
    print("Error: requests package not installed. Run: pip install requests")
    sys.exit(1)

try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets package not installed. Run: pip install datasets")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Error: tqdm package not installed. Run: pip install tqdm")
    sys.exit(1)


def call_qwen_openrouter(
    messages: List[Dict[str, str]],
    api_key: str,
    model: str = "qwen/qwen-2.5-72b-instruct",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    max_retries: int = 5
) -> Optional[str]:
    """
    Call Qwen model via OpenRouter API with retry logic.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        api_key: OpenRouter API key
        model: Model identifier for OpenRouter
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        max_retries: Maximum number of retry attempts

    Returns:
        Generated response text, or None if all retries failed
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": messages
                },
                timeout=60
            )

            response.raise_for_status()
            data = response.json()

            # Extract generated content
            generated_content = data['choices'][0]['message']['content']
            return generated_content

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  Timeout on attempt {attempt + 1}/{max_retries}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts (timeout)")
                return None

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  API error on attempt {attempt + 1}/{max_retries}: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_data = e.response.json()
                        print(f"  Error details: {error_data}")
                    except:
                        print(f"  Response text: {e.response.text[:200]}")
                print(f"  Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return None

        except Exception as e:
            print(f"  Unexpected error: {e}")
            return None

    return None


def extract_messages_from_dolci_sample(sample: Dict) -> Optional[List[Dict[str, str]]]:
    """
    Extract messages from a DOLCI dataset sample.

    The DOLCI dataset may have different formats. This function handles:
    - 'messages' field (list of message dicts)
    - 'conversations' field (list of message dicts)
    - 'prompt' and 'response' fields

    Returns the messages list excluding the final assistant response (which we'll regenerate).
    """
    messages = None

    if 'messages' in sample:
        messages = sample['messages']
    elif 'conversations' in sample:
        messages = sample['conversations']
    elif 'prompt' in sample and 'response' in sample:
        # Convert to messages format
        messages = [
            {"role": "user", "content": sample['prompt']}
        ]
    else:
        return None

    # Remove the last assistant message if present (we'll regenerate it)
    if messages and len(messages) > 0 and messages[-1].get('role') == 'assistant':
        return messages[:-1]

    return messages


def process_single_sample(
    idx: int,
    sample: Dict,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int
) -> Optional[Dict]:
    """
    Process a single sample and generate a response.

    Args:
        idx: Sample index
        sample: DOLCI dataset sample
        api_key: OpenRouter API key
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Result dictionary with index, messages, and responses, or None if failed
    """
    # Extract messages (excluding final assistant response)
    messages = extract_messages_from_dolci_sample(sample)

    if messages is None:
        return None

    # Get original response (for comparison/debugging)
    original_response = None
    if 'messages' in sample and len(sample['messages']) > 0:
        if sample['messages'][-1].get('role') == 'assistant':
            original_response = sample['messages'][-1].get('content')
    elif 'conversations' in sample and len(sample['conversations']) > 0:
        if sample['conversations'][-1].get('role') == 'assistant':
            original_response = sample['conversations'][-1].get('content')
    elif 'response' in sample:
        original_response = sample['response']

    # Generate new response with Qwen
    qwen_response = call_qwen_openrouter(
        messages=messages,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )

    if qwen_response is None:
        return None

    # Store result
    return {
        "index": idx,
        "messages": messages,
        "original_response": original_response,
        "qwen_response": qwen_response
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate Qwen 2.5 Instruct responses for DOLCI dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate responses for first 1000 samples with 100 concurrent workers
  python scripts/generate_qwen_dolci_responses.py --output dolci_qwen_responses.json --num-samples 1000

  # Generate responses for all samples with 50 concurrent workers
  python scripts/generate_qwen_dolci_responses.py --output dolci_qwen_responses.json --num-samples -1 --max-workers 50

  # Use specific Qwen model
  python scripts/generate_qwen_dolci_responses.py --output dolci_qwen_responses.json --model qwen/qwen-2.5-7b-instruct
        """
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dolci_qwen_responses.json",
        help="Output file path for saved responses (default: dolci_qwen_responses.json)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen/qwen-2.5-72b-instruct",
        help="Qwen model to use via OpenRouter (default: qwen/qwen-2.5-72b-instruct)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="Number of samples to process (-1 for all, default: 5000)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate per response (default: 2048)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file if it exists"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=100,
        help="Maximum number of concurrent workers (default: 100)"
    )

    args = parser.parse_args()

    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)

    print(f"Loading DOLCI-Instruct-SFT dataset...")
    try:
        dolci_dataset = load_dataset("allenai/Dolci-Instruct-SFT", split="train")
        print(f"Loaded {len(dolci_dataset)} samples from DOLCI dataset")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Determine number of samples to process
    num_samples = args.num_samples if args.num_samples > 0 else len(dolci_dataset)
    num_samples = min(num_samples, len(dolci_dataset))

    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Samples to process: {num_samples}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Max workers: {args.max_workers}")
    print(f"  Output file: {args.output}")
    print()

    # Load existing responses if resuming
    existing_responses = {}
    start_idx = 0

    if args.resume and Path(args.output).exists():
        print(f"Resuming from existing output file: {args.output}")
        try:
            with open(args.output, 'r') as f:
                existing_data = json.load(f)
                existing_responses = {item['index']: item for item in existing_data}
                start_idx = len(existing_responses)
                print(f"Found {start_idx} existing responses, starting from index {start_idx}")
        except Exception as e:
            print(f"Warning: Could not load existing file: {e}")
            print("Starting from scratch...")

    # Generate responses
    results = []
    failed_indices = []

    # Add existing responses to results
    for idx in range(start_idx):
        if idx in existing_responses:
            results.append(existing_responses[idx])

    print(f"Generating responses for samples {start_idx} to {num_samples} using {args.max_workers} workers...")
    print()

    # Thread-safe lock for checkpointing
    checkpoint_lock = Lock()
    completed_count = start_idx

    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_idx = {}
        for idx in range(start_idx, num_samples):
            sample = dolci_dataset[idx]
            future = executor.submit(
                process_single_sample,
                idx,
                sample,
                api_key,
                args.model,
                args.temperature,
                args.max_tokens
            )
            future_to_idx[future] = idx

        # Process completed tasks
        with tqdm(total=num_samples - start_idx, desc="Processing samples") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]

                try:
                    result = future.result()

                    if result is None:
                        failed_indices.append(idx)
                    else:
                        results.append(result)

                    completed_count += 1

                    # Save checkpoint every 100 completed samples
                    if completed_count % 100 == 0:
                        with checkpoint_lock:
                            # Sort results by index before saving
                            sorted_results = sorted(results, key=lambda x: x['index'])
                            with open(args.output, 'w') as f:
                                json.dump(sorted_results, f, indent=2)
                            tqdm.write(f"Checkpoint: Saved {len(results)} responses to {args.output}")

                except Exception as e:
                    tqdm.write(f"Error processing sample {idx}: {e}")
                    failed_indices.append(idx)

                pbar.update(1)

    # Final save
    print(f"\nSaving final results to {args.output}...")
    sorted_results = sorted(results, key=lambda x: x['index'])
    with open(args.output, 'w') as f:
        json.dump(sorted_results, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples processed: {num_samples}")
    print(f"Successful generations: {len(results)}")
    print(f"Failed generations: {len(failed_indices)}")
    if failed_indices:
        print(f"Failed indices: {failed_indices[:20]}" + ("..." if len(failed_indices) > 20 else ""))
    print(f"Output saved to: {args.output}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
