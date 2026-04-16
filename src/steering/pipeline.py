"""Steering vector pipeline.

Implements difference-of-means steering vectors with effect size
and collateral damage measurement.
"""

import torch
import numpy as np
from dataclasses import dataclass, field


@dataclass
class SteeringResult:
    """Result of a steering experiment for one concept at one layer."""
    concept: str
    family: str
    layer: int

    # Steering vector
    steering_direction: np.ndarray | None = None  # unit norm
    steering_magnitude: float = 0.0

    # Per-prompt metrics
    effect_sizes: np.ndarray = field(default_factory=lambda: np.array([]))
    collateral_generic: float = 0.0  # KL on unrelated prompts
    collateral_concept: dict[str, float] = field(default_factory=dict)  # ΔA_lin per other family

    # Aggregate
    mean_effect_size: float = 0.0
    success: bool = False  # effect_size > 0.1 and low collateral

    @property
    def effect_size_per_norm(self) -> float:
        if self.steering_magnitude < 1e-10:
            return 0.0
        return self.mean_effect_size / self.steering_magnitude


class SteeringPipeline:
    """Steering vector extraction and evaluation.

    Computes difference-of-means steering vectors and measures
    their effectiveness and collateral damage.
    """

    def __init__(self, extractor, alpha_range: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)):
        """
        Args:
            extractor: ActivationExtractor instance.
            alpha_range: Scaling factors to test for steering strength.
        """
        self.extractor = extractor
        self.alpha_range = alpha_range

    def compute_steering_vector(
        self,
        positive_activations: np.ndarray,
        negative_activations: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Compute steering vector as difference of means.

        Args:
            positive_activations: (n_pos, d_model) activations for positive examples.
            negative_activations: (n_neg, d_model) activations for negative examples.

        Returns:
            (direction, magnitude): Unit-norm direction and original magnitude.
        """
        diff = positive_activations.mean(axis=0) - negative_activations.mean(axis=0)
        magnitude = float(np.linalg.norm(diff))
        if magnitude < 1e-10:
            return diff, 0.0
        direction = diff / magnitude
        return direction, magnitude

    @torch.no_grad()
    def evaluate_steering(
        self,
        prompts: list[str],
        target_token_ids: list[int],
        steering_direction: np.ndarray,
        layer: int,
        alpha: float = 1.0,
        batch_size: int = 8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate steering effect on target behavior.

        Args:
            prompts: Prompts to steer.
            target_token_ids: Token IDs for desired behavior per prompt.
            steering_direction: Unit-norm steering direction.
            layer: Layer to inject steering.
            alpha: Scaling factor.
            batch_size: Batch size.

        Returns:
            (effect_sizes, baseline_probs): Per-prompt effect sizes and baselines.
        """
        n = len(prompts)

        # Baseline probabilities
        baseline_probs_all = self.extractor.get_next_token_probs(prompts, batch_size=batch_size)

        # Steered probabilities
        perturbation = torch.tensor(
            np.tile(steering_direction, (n, 1)) * alpha,
            dtype=torch.float32,
        )
        steered_logits = self.extractor.extract_with_perturbation(
            prompts, layer, perturbation, batch_size=batch_size
        )
        steered_probs = torch.softmax(steered_logits.float(), dim=-1)

        # Effect size per prompt
        effect_sizes = np.zeros(n)
        baseline_target_probs = np.zeros(n)
        for i in range(n):
            tid = target_token_ids[i]
            p_base = baseline_probs_all[i, tid].item()
            p_steer = steered_probs[i, tid].item()
            effect_sizes[i] = p_steer - p_base
            baseline_target_probs[i] = p_base

        return effect_sizes, baseline_target_probs

    @torch.no_grad()
    def measure_collateral_generic(
        self,
        unrelated_prompts: list[str],
        steering_direction: np.ndarray,
        layer: int,
        alpha: float = 1.0,
        batch_size: int = 8,
    ) -> float:
        """Measure generic collateral damage on unrelated prompts.

        Returns mean KL divergence between steered and unsteered output distributions.
        """
        n = len(unrelated_prompts)

        # Baseline
        baseline_probs = self.extractor.get_next_token_probs(
            unrelated_prompts, batch_size=batch_size
        )

        # Steered
        perturbation = torch.tensor(
            np.tile(steering_direction, (n, 1)) * alpha,
            dtype=torch.float32,
        )
        steered_logits = self.extractor.extract_with_perturbation(
            unrelated_prompts, layer, perturbation, batch_size=batch_size
        )
        steered_probs = torch.softmax(steered_logits.float(), dim=-1)

        # KL divergence per prompt
        # KL(baseline || steered) = sum(baseline * log(baseline / steered))
        eps = 1e-10
        kl_divs = []
        for i in range(n):
            p = baseline_probs[i].numpy() + eps
            q = steered_probs[i].numpy() + eps
            kl = float(np.sum(p * np.log(p / q)))
            kl_divs.append(kl)

        return float(np.mean(kl_divs))

    def run(
        self,
        positive_activations: np.ndarray,
        negative_activations: np.ndarray,
        test_prompts: list[str],
        target_token_ids: list[int],
        unrelated_prompts: list[str],
        layer: int,
        concept: str,
        family: str,
        alpha: float = 1.0,
        batch_size: int = 8,
    ) -> SteeringResult:
        """Run full steering evaluation.

        Args:
            positive_activations: Activations for positive examples.
            negative_activations: Activations for negative examples.
            test_prompts: Prompts to steer (can be different from training).
            target_token_ids: Target token IDs per test prompt.
            unrelated_prompts: Prompts for collateral measurement.
            layer: Injection layer.
            concept: Concept name.
            family: Family name.
            alpha: Steering strength.
            batch_size: Batch size.
        """
        direction, magnitude = self.compute_steering_vector(
            positive_activations, negative_activations
        )

        effect_sizes, _ = self.evaluate_steering(
            test_prompts, target_token_ids, direction, layer, alpha, batch_size
        )

        collateral = self.measure_collateral_generic(
            unrelated_prompts, direction, layer, alpha, batch_size
        )

        mean_effect = float(np.mean(effect_sizes))

        return SteeringResult(
            concept=concept,
            family=family,
            layer=layer,
            steering_direction=direction,
            steering_magnitude=magnitude,
            effect_sizes=effect_sizes,
            collateral_generic=collateral,
            mean_effect_size=mean_effect,
            success=(abs(mean_effect) > 0.1),
        )
