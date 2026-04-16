"""Default configuration for LAP experiments."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name: str = "google/gemma-2-2b"
    device: str = "cuda"
    dtype: str = "float16"
    cache_dir: str | None = None


@dataclass
class DataConfig:
    n_pairs: int = 100
    use_sentiment_fallback: bool = False
    cache_dir: str | None = None
    max_tokens: int = 64


@dataclass
class ProbeConfig:
    n_folds: int = 5
    n_permutations: int = 5
    sample_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    pca_dims: tuple[int, ...] = (16, 32, 64, 128)
    mlp_hidden: int = 256
    mlp_dropout: float = 0.2
    mlp_epochs: int = 100
    mlp_patience: int = 10


@dataclass
class LAPConfig:
    n_perturbations: int = 10
    perturbation_alpha: float = 0.01
    condition_number_pcs: int = 15
    pca_k_values: tuple[int, ...] = (32, 128)


@dataclass
class SteeringConfig:
    alpha_range: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    default_alpha: float = 1.0


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    lap: LAPConfig = field(default_factory=LAPConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)

    # Experiment control
    batch_size: int = 8
    results_dir: str = "results"
    seed: int = 42

    # Which experiments to run
    run_exp1: bool = True  # LAP across layers
    run_exp2: bool = True  # LAP predicts tool success
    run_exp3: bool = True  # Failure clustering
    run_exp4: bool = True  # Chaotic regime
    run_exp5: bool = False  # Post-training (off by default)

    # Experiment-specific
    exp2_n_prompts: int = 300  # Prompts for attribution graphs
    exp5_base_model: str = "meta-llama/Llama-3.2-1B"
    exp5_instruct_model: str = "meta-llama/Llama-3.2-1B-Instruct"
