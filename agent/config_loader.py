"""Configuration loader for attack-specific settings.

This module provides a unified way to load and access all attack-specific
configuration including behavior descriptions, judge prompts, model settings,
and hyperparameters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class AttackMetadata:
    """Attack metadata including name, key, and description."""
    name: str
    key: str
    description: str


@dataclass
class SeedExample:
    """Seed example for hypothesis generation.

    Supports two formats:
    1. prompt format: evaluator_prompt (str) and target_response (str)
    2. conversation format: conversation (list of message dicts) and target_continuation (str)
    """
    format: str  # "prompt" or "conversation"
    # For prompt format
    evaluator_prompt: Optional[str] = None
    target_response: Optional[str] = None
    # For conversation format
    conversation: Optional[List[Dict[str, str]]] = None
    target_continuation: Optional[str] = None

    def __post_init__(self):
        """Validate that required fields are present based on format."""
        if self.format == "prompt":
            if not self.evaluator_prompt or not self.target_response:
                raise ValueError("prompt format requires evaluator_prompt and target_response")
        elif self.format == "conversation":
            if not self.conversation or not self.target_continuation:
                raise ValueError("conversation format requires conversation and target_continuation")
        else:
            raise ValueError(f"Invalid format: {self.format}. Must be 'prompt' or 'conversation'")


@dataclass
class JudgePrompt:
    """Judge prompt configuration with system and user templates."""
    system: str
    user_template: str

    def format_user_prompt(self, prompt: str, response: str) -> str:
        """Format the user prompt template with actual prompt and response."""
        return self.user_template.format(prompt=prompt, response=response or "(no response)")


@dataclass
class JudgePrompts:
    """Collection of judge prompts for evaluation."""
    harmful_detection: JudgePrompt
    refusal_detection: JudgePrompt


@dataclass
class ConversationFormat:
    """Conversation format settings."""
    output_format: str  # "prompt" or "conversation"
    num_turns: int
    include_system_prompt: bool


@dataclass
class TargetModelConfig:
    """Target model configuration."""
    base_url: str
    model_name: str
    max_tokens: int
    temperature: float
    request_timeout: int
    icl_eval_model: str

    def get_model_name_for_files(self) -> str:
        """Get a clean model name for use in filenames."""
        name = self.model_name.split("/")[-1]
        name = name.replace("/", "-").replace("_", "-")
        return name


@dataclass
class JudgeModelConfig:
    """Judge model configuration."""
    model_name: str
    max_tokens: int
    temperature: float


@dataclass
class HypothesisGenerationConfig:
    """Hypothesis generation settings."""
    max_ideas: int
    max_concrete_examples: int
    max_variations_per_example: int
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int


@dataclass
class DataGenerationConfig:
    """Data generation settings for bloom-evals."""
    num_variations: int
    parallel_requests: int
    max_concurrent: int = 20  # Default to 20 for backward compatibility
    benign_generation_prompt: Optional[str] = None  # Custom prompt for generating benign counterparts


@dataclass
class ExpectedBehaviorPrompts:
    """Prompts for generating expected model behaviors during training data creation."""
    harmful_system_prompt: str  # System prompt for generating response to harmful prompts
    harmful_user_template: str  # User template for harmful prompts (use {prompt} placeholder)
    benign_system_prompt: str   # System prompt for generating response to benign prompts
    benign_user_template: str   # User template for benign prompts (use {prompt} placeholder)
    harmful_default_response: str  # Fallback response if generation fails for harmful
    benign_default_response: str   # Fallback response if generation fails for benign


@dataclass
class PathsConfig:
    """File paths configuration."""
    behaviors_file: str
    dolci_responses_file: Optional[str] = None  # Path to pre-generated DOLCI responses for SFT mixing


@dataclass
class EvaluationConfig:
    """Evaluation settings."""
    train_split: float
    val_split: float
    test_split: float
    random_seed: int


@dataclass
class AttackConfig:
    """Unified attack configuration containing all settings."""
    attack: AttackMetadata
    seed_example: SeedExample
    judge_prompts: JudgePrompts
    conversation_format: ConversationFormat
    target_model: TargetModelConfig
    judge_model: JudgeModelConfig
    hypothesis_generation: HypothesisGenerationConfig
    data_generation: DataGenerationConfig
    paths: PathsConfig
    evaluation: Optional[EvaluationConfig] = None  # Optional, not used by evaluation code
    expected_behavior_prompts: Optional[ExpectedBehaviorPrompts] = None  # Optional, for generating expected behaviors

    @classmethod
    def from_file(cls, config_path: str | Path) -> AttackConfig:
        """Load attack configuration from a JSON or YAML file.

        Args:
            config_path: Path to the configuration file (.json or .yaml/.yml)

        Returns:
            AttackConfig object with all settings loaded

        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If file format is unsupported or config is malformed
            KeyError: If required config sections are missing
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load based on file extension
        suffix = config_path.suffix.lower()
        if suffix == '.json':
            with open(config_path, 'r') as f:
                data = json.load(f)
        elif suffix in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {suffix}. Use .json, .yaml, or .yml")

        if not data:
            raise ValueError(f"Empty config file: {config_path}")

        # Parse each section
        attack = AttackMetadata(**data['attack'])

        # Handle seed_example - add format if not present for backward compatibility
        seed_example_data = data['seed_example'].copy()
        if 'format' not in seed_example_data:
            # Old format: assume prompt format
            seed_example_data['format'] = 'prompt'
        seed_example = SeedExample(**seed_example_data)

        judge_prompts = JudgePrompts(
            harmful_detection=JudgePrompt(**data['judge_prompts']['harmful_detection']),
            refusal_detection=JudgePrompt(**data['judge_prompts']['refusal_detection'])
        )

        conversation_format = ConversationFormat(**data['conversation_format'])

        target_model = TargetModelConfig(**data['target_model'])

        judge_model = JudgeModelConfig(**data['judge_model'])

        hypothesis_generation = HypothesisGenerationConfig(**data['hypothesis_generation'])

        data_generation = DataGenerationConfig(**data['data_generation'])

        paths = PathsConfig(**data['paths'])

        # Evaluation section is optional (not used by evaluation code)
        evaluation = None
        if 'evaluation' in data:
            evaluation = EvaluationConfig(**data['evaluation'])

        # Expected behavior prompts section is optional
        expected_behavior_prompts = None
        if 'expected_behavior_prompts' in data:
            expected_behavior_prompts = ExpectedBehaviorPrompts(**data['expected_behavior_prompts'])

        return cls(
            attack=attack,
            seed_example=seed_example,
            judge_prompts=judge_prompts,
            conversation_format=conversation_format,
            target_model=target_model,
            judge_model=judge_model,
            hypothesis_generation=hypothesis_generation,
            data_generation=data_generation,
            paths=paths,
            evaluation=evaluation,
            expected_behavior_prompts=expected_behavior_prompts
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format."""
        return {
            'attack': {
                'name': self.attack.name,
                'key': self.attack.key,
                'description': self.attack.description
            },
            'seed_example': {
                'evaluator_prompt': self.seed_example.evaluator_prompt,
                'target_response': self.seed_example.target_response
            },
            'judge_prompts': {
                'harmful_detection': {
                    'system': self.judge_prompts.harmful_detection.system,
                    'user_template': self.judge_prompts.harmful_detection.user_template
                },
                'refusal_detection': {
                    'system': self.judge_prompts.refusal_detection.system,
                    'user_template': self.judge_prompts.refusal_detection.user_template
                }
            },
            'conversation_format': {
                'output_format': self.conversation_format.output_format,
                'num_turns': self.conversation_format.num_turns,
                'include_system_prompt': self.conversation_format.include_system_prompt
            },
            'target_model': {
                'base_url': self.target_model.base_url,
                'model_name': self.target_model.model_name,
                'max_tokens': self.target_model.max_tokens,
                'temperature': self.target_model.temperature,
                'request_timeout': self.target_model.request_timeout,
                'icl_eval_model': self.target_model.icl_eval_model
            },
            'judge_model': {
                'model_name': self.judge_model.model_name,
                'max_tokens': self.judge_model.max_tokens,
                'temperature': self.judge_model.temperature
            },
            'hypothesis_generation': {
                'max_ideas': self.hypothesis_generation.max_ideas,
                'max_concrete_examples': self.hypothesis_generation.max_concrete_examples,
                'max_variations_per_example': self.hypothesis_generation.max_variations_per_example,
                'llm_model': self.hypothesis_generation.llm_model,
                'llm_temperature': self.hypothesis_generation.llm_temperature,
                'llm_max_tokens': self.hypothesis_generation.llm_max_tokens
            },
            'data_generation': {
                'num_variations': self.data_generation.num_variations,
                'parallel_requests': self.data_generation.parallel_requests
            },
            'paths': {
                'behaviors_file': self.paths.behaviors_file,
                'scenario_file': self.paths.scenario_file
            },
            'evaluation': {
                'train_split': self.evaluation.train_split,
                'val_split': self.evaluation.val_split,
                'test_split': self.evaluation.test_split,
                'random_seed': self.evaluation.random_seed
            }
        }


def load_attack_config(config_path: str | Path) -> AttackConfig:
    """Convenience function to load attack configuration from JSON or YAML file.

    Args:
        config_path: Path to the configuration file (.json, .yaml, or .yml)

    Returns:
        AttackConfig object with all settings loaded
    """
    return AttackConfig.from_file(config_path)


__all__ = [
    "AttackConfig",
    "AttackMetadata",
    "SeedExample",
    "JudgePrompt",
    "JudgePrompts",
    "ConversationFormat",
    "TargetModelConfig",
    "JudgeModelConfig",
    "HypothesisGenerationConfig",
    "DataGenerationConfig",
    "ExpectedBehaviorPrompts",
    "PathsConfig",
    "EvaluationConfig",
    "load_attack_config"
]
