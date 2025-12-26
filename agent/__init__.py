"""A3 Agent module for hypothesis-driven safety research."""

from .experiment_log import ExperimentLog, ExperimentEntry
from .config_loader import (
    AttackConfig,
    AttackMetadata,
    SeedExample,
    JudgePrompt,
    JudgePrompts,
    ConversationFormat,
    TargetModelConfig,
    JudgeModelConfig,
    HypothesisGenerationConfig,
    DataGenerationConfig,
    ExpectedBehaviorPrompts,
    PathsConfig,
    EvaluationConfig,
    load_attack_config,
)
from .research_hypothesis_agent import ResearchHypothesisAgent, AgentConfig
from .data_generation_agent import DataGenerationAgent, DataGenConfig
from .test_target_model import TestTargetModel, TargetModelConfig, JudgeConfig
from .evaluation_agent import EvaluationAgent, EvalAgentConfig, DataSplit, EvaluationResults
from .benchmark_evaluator import BenchmarkEvaluator, BenchmarkConfig, BenchmarkResults
from .sft_agent import SFTAgent, SFTConfig, SFTResult
from .iterative_sft_lora_agent import (
    IterativeLoRASFTAgent,
    HypothesisWeights,
    HyperparametersResponse,
    IterativeSFTResult,
)
from .icl_defense_agent import ICLDefenseAgent, ICLDefenseConfig

# DSPy is optional - only import if dspy is installed
try:
    from .dspy_defense_agent import DSPyDefenseAgent, DSPyDefenseConfig
    _DSPY_AVAILABLE = True
except (ImportError, RuntimeError):
    DSPyDefenseAgent = None
    DSPyDefenseConfig = None
    _DSPY_AVAILABLE = False

__all__ = [
    # Experiment Log
    "ExperimentLog",
    "ExperimentEntry",
    # Config Loader
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
    "load_attack_config",
    # Research Hypothesis Agent
    "ResearchHypothesisAgent",
    "AgentConfig",
    # Data Generation Agent
    "DataGenerationAgent",
    "DataGenConfig",
    # Test Target Model
    "TestTargetModel",
    "JudgeConfig",
    # Evaluation Agent
    "EvaluationAgent",
    "EvalAgentConfig",
    "DataSplit",
    "EvaluationResults",
    # Benchmark Evaluator
    "BenchmarkEvaluator",
    "BenchmarkConfig",
    "BenchmarkResults",
    # SFT Agent
    "SFTAgent",
    "SFTConfig",
    "SFTResult",
    # Iterative LoRA SFT Agent
    "IterativeLoRASFTAgent",
    "HypothesisWeights",
    "HyperparametersResponse",
    "IterativeSFTResult",
    # DSPy Defense Agent
    "DSPyDefenseAgent",
    "DSPyDefenseConfig",
    # ICL Defense Agent
    "ICLDefenseAgent",
    "ICLDefenseConfig",
]
