"""
Step 1: Data Generation
-----------------------
Runs hypothesis generation and testing rounds for behavior-based safety research.
Generates attack prompts and benign counterparts for training data.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on the path for package imports
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.research_hypothesis_agent import ResearchHypothesisAgent  # noqa: E402
from agent.config_loader import load_attack_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run hypothesis generation and testing rounds for behavior-based safety research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with sycophancy config for Llama
  python scripts/step1_data_generation.py --config-file configs/sycophancy-llama.json

  # Use a different model for hypothesis generation
  python scripts/step1_data_generation.py --config-file configs/sycophancy-llama.json \\
    --model claude-opus-4-20250514

  # Run 20 rounds with 50 samples per hypothesis
  python scripts/step1_data_generation.py --config-file configs/sycophancy-llama.json \\
    --num-rounds 20 --num-samples 50
        """
    )
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to attack config JSON file (e.g., configs/sycophancy-llama.json)"
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=10,
        help="Number of hypothesis generation rounds (default: 10)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of prompts to generate per hypothesis (default: 20)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Model to use for hypothesis/data generation (default: claude-sonnet-4-20250514)"
    )
    args = parser.parse_args()

    # Require Anthropic API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "Environment variable ANTHROPIC_API_KEY is required"
        )

    print(f"Loading attack config from: {args.config_file}")
    attack_config = load_attack_config(args.config_file)

    # Create agent with specified model
    agent = ResearchHypothesisAgent(attack_config=attack_config, hypothesis_model=args.model)

    # Print configuration
    print(f"Attack: {attack_config.attack.name} ({attack_config.attack.key})")
    print(f"Target model: {attack_config.target_model.model_name}")
    print(f"Judge model: {attack_config.judge_model.model_name}")
    print(f"Hypothesis generation model: {args.model}")
    print(f"Output format: {attack_config.conversation_format.output_format}")
    if attack_config.conversation_format.output_format == "conversation":
        print(f"  Conversation turns: {attack_config.conversation_format.num_turns}")
        print(f"  Include system prompt: {attack_config.conversation_format.include_system_prompt}")

    num_rounds = args.num_rounds
    num_samples = args.num_samples

    print(f"\n{'='*100}")
    print(f"RUNNING {num_rounds} ROUNDS OF HYPOTHESIS GENERATION")
    print(f"{'='*100}\n")

    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*100}")
        print(f"ROUND {round_num}/{num_rounds}")
        print(f"{'='*100}\n")

        # Propose hypotheses for this round
        start_idx, end_idx, hypotheses = agent.propose_hypotheses()
        print(f"Proposed {len(hypotheses)} hypotheses for round {round_num}")
        print("="*100)

        # Generate data and test each hypothesis
        all_results = agent.generate_data_for_hypotheses(start_idx, end_idx, hypotheses, num_samples=num_samples)

        # Print summary for this round
        print(f"\n{'='*100}")
        print(f"ROUND {round_num} SUMMARY")
        print(f"{'='*100}")
        for result in all_results:
            stats = result['stats']
            print(f"  H {result['hypothesis_index'] + 1}:")
            print(f"    Harmful: {stats['harmful_success_count']}/{stats['harmful_total']} attack success ({stats['harmful_success_rate']:.1%})")
            print(f"    Benign:  {stats['benign_refusal_count']}/{stats['benign_total']} refusals ({stats['benign_refusal_rate']:.1%})")
            print(f"    Text: {result['hypothesis_text'][:80]}...")
        print(f"{'='*100}\n")

    # Print final log state
    print("\n=== Final Experiment Log (tail) ===\n")
    print("\n".join(agent.log.as_text().splitlines()))


if __name__ == "__main__":
    main()
