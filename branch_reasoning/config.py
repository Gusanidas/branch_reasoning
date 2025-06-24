from omegaconf import OmegaConf
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from enum import Enum
import torch
import yaml


class BranchedCompletionScoringMethod(Enum):
    MAX = "max"
    AVERAGE = "average"


class NormalizeMethod(Enum):
    PROMPT = "prompt"
    ALL = "all"
    BOTH = "both"


@dataclass
class GenerationConfig:
    """Configuration for text generation parameters."""

    # In one iteration, how many solutions to generate. Several branches can belong to the same completion.
    total_completions: int = 1536
    # How many completions to generate in one batch.
    gen_batch_size: int = 256
    # How many completions to generate per prompt.
    no_completions: int = 8
    branch_completions: bool = False
    # How many branches to generate for each branching point.
    branching_factor: int = 3
    # Maximum number of branching points.
    max_branching_points: int = 2
    # Maximum number of concurrent requests to the vLLM server.
    max_concurrent_requests: Optional[int] = None
    # Maximum length of the completion.
    max_len: int = 1536
    # Whether to use the vLLM server.
    use_vllm: bool = True
    use_examples: bool = True
    temperature: float = 1.0
    opt_generation_args: Dict[str, Any] = field(
        default_factory=lambda: {
            "top_p": 0.9,
        }
    )


@dataclass
class ModelConfig:
    """Configuration for model and tokenizer settings."""

    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    tokenizer_name: Optional[str] = None
    reference_model_name: Optional[str] = "Qwen/Qwen2.5-Coder-3B-Instruct"
    device: str = "cuda"
    use_bfloat16: bool = True

    def __post_init__(self):
        # Auto-set tokenizer_name if not provided
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    # Number of epochs per iteration.
    num_epochs: int = 2
    epsilon_low: float = 0.1
    epsilon_high: float = 0.1
    max_grad_norm: float = 2.5
    # Regularization parameter. KL divergence with reference model.
    beta: float = 0.005
    learning_rate: float = 5e-6
    weight_decay: float = 0.00001
    gradient_accumulation_steps: int = 128
    max_gradient_accumulation_steps: int = 128
    # Batch size for training. Per GPU.
    batch_size: int = 3
    total_steps: int = 250
    warmup_steps: int = 16
    log_probs_batch_size: int = 4
    full_last_epoch: bool = False
    max_grad_threshold: float = 22.0
    clip_log_ratio_value: int = 4
    optimizer_path: Optional[str] = "optimizer.pt"
    scheduler_path: Optional[str] = "scheduler.pt"
    # Enable torch.compile for performance optimization
    use_torch_compile: bool = False
    # Enable gradient checkpointing
    use_gradient_checkpointing: bool = False


@dataclass
class VLLMServerConfig:
    """Configuration for VLLM server settings."""

    tensor_parallel_size: int = 1
    data_parallel_size: int = 2
    gpu_memory_utilization: float = 0.8
    max_model_len: int = 1536


@dataclass
class ScoringConfig:
    """Configuration for scoring variables."""

    normalize_by_prompt: NormalizeMethod = NormalizeMethod.BOTH
    normalize_prompt_ratio: float = 0.8
    normalize_all_branches: bool = False
    scoring_method: BranchedCompletionScoringMethod = (
        BranchedCompletionScoringMethod.AVERAGE
    )


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking and data."""

    save_interval: int = 1000
    hf_repo_name: str = "Gusanidas/branch-grpo-model-qwen-3b-branch3"
    hf_dataset_name: str = "Gusanidas/countdown-tasks-dataset-med-vl6"
    iterations: int = 100
    wandb_logging: bool = True
    wandb_project_name: str = "branch_grpo_vast_branch_vllm"
    run_name: str = "run_706"
    reuse_completions: bool = False
    reuse_completions_start_iter: int = 4
    reuse_completions_batch_size: int = 24
    reuse_completions_recent_batch_size: int = 24
    model_dir: Optional[str] = "model"
    ref_model_dir: Optional[str] = "ref_model"
    data_path: Optional[str] = "completions.jsonl"
    training_log_path: Optional[str] = "training_logs.jsonl"


@dataclass
class BranchGRPOConfig:
    """Main configuration class containing all sub-configurations."""

    generation: GenerationConfig = field(default_factory=GenerationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    vllm_server: VLLMServerConfig = field(default_factory=VLLMServerConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    verbose: bool = False
    iteration: Optional[int] = 0

    def sync_vllm_max_length_with_generation(self):
        """Set vLLM server max_model_len to match the generation max_len."""
        self.vllm_server.max_model_len = self.generation.max_len


if __name__ == "__main__":
    config = BranchGRPOConfig()

    omega_config = OmegaConf.structured(config)
    print(OmegaConf.to_yaml(omega_config))

    # Write configuration to a YAML file
    config_file = "H100_3b_config.yaml"
    with open(config_file, "w") as f:
        f.write(OmegaConf.to_yaml(omega_config))
    print(f"\nConfiguration written to {config_file}")
