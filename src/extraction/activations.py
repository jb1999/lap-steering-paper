"""Activation extraction pipeline.

Extracts residual stream activations at specified token positions
across all layers of a transformer model.
"""

import torch
import numpy as np
from typing import Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ExtractionResult:
    """Result of activation extraction for a batch of prompts.

    Attributes:
        activations: Dict mapping layer index to tensor of shape (n_prompts, d_model).
        token_positions: The token positions that were extracted.
        prompts: The original prompt strings.
    """
    activations: dict[int, torch.Tensor]  # layer -> (n_prompts, d_model)
    token_positions: list[int]
    prompts: list[str]

    def to_numpy(self) -> dict[int, np.ndarray]:
        return {l: a.cpu().float().numpy() for l, a in self.activations.items()}

    def save(self, path: str):
        torch.save({
            "activations": {l: a.cpu() for l, a in self.activations.items()},
            "token_positions": self.token_positions,
            "prompts": self.prompts,
        }, path)

    @classmethod
    def load(cls, path: str) -> "ExtractionResult":
        data = torch.load(path, weights_only=False)
        return cls(**data)


class ActivationExtractor:
    """Extract residual stream activations from a transformer.

    Usage:
        extractor = ActivationExtractor("google/gemma-2-2b")
        result = extractor.extract(["The capital of France is"], layers=[0, 5, 10])
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        cache_dir: str | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-padding for causal LMs — ensures last real token is at
        # the right edge, preventing padding from corrupting batched inference
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            cache_dir=cache_dir,
        ).to(device)
        self.model.eval()

        self.n_layers = self.model.config.num_hidden_layers
        self.d_model = self.model.config.hidden_size

    def get_prefix_positions(self, prompts: list[str], prefixes: list[str]) -> list[int]:
        """Compute token position of the last token in each prefix.

        For contrastive pairs that share a prefix and differ at the end,
        this returns the position to extract at (last prefix token).

        Args:
            prompts: Full prompt strings (prefix + answer).
            prefixes: Shared prefix strings.

        Returns:
            List of token positions (one per prompt).
        """
        positions = []
        for prefix in prefixes:
            prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
            positions.append(len(prefix_ids) - 1)  # last token of prefix
        return positions

    @torch.no_grad()
    def extract(
        self,
        prompts: list[str],
        layers: list[int] | None = None,
        token_position: str = "last",
        extraction_positions: list[int] | None = None,
        batch_size: int = 8,
    ) -> ExtractionResult:
        """Extract residual stream activations.

        Args:
            prompts: List of prompt strings.
            layers: Layer indices to extract from. None = all layers.
            token_position: "last" for last non-padding token.
            extraction_positions: Per-prompt token positions to extract at.
                Overrides token_position if provided.
            batch_size: Batch size for inference.

        Returns:
            ExtractionResult with activations per layer.
        """
        if layers is None:
            layers = list(range(self.n_layers))

        all_activations = {l: [] for l in layers}
        all_positions = []

        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start:batch_start + batch_size]

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.device)

            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            # hidden_states: tuple of (n_layers + 1) tensors, each (batch, seq, d_model)
            # Index 0 is embedding output, index i is output of layer i-1
            hidden_states = outputs.hidden_states

            # Determine extraction positions
            if extraction_positions is not None:
                batch_positions = extraction_positions[batch_start:batch_start + len(batch_prompts)]
                # Account for any BOS token the tokenizer adds
                bos_offset = 1 if self.tokenizer.bos_token_id is not None else 0
                positions = torch.tensor(
                    [p + bos_offset for p in batch_positions],
                    device=self.device,
                )
            elif token_position == "last":
                # With left-padding, last real token is always at position -1
                seq_len = inputs["input_ids"].shape[1]
                positions = torch.full(
                    (len(batch_prompts),), seq_len - 1, device=self.device
                )
            else:
                positions = torch.full(
                    (len(batch_prompts),), -1, device=self.device
                )

            all_positions.extend(positions.cpu().tolist())

            # Extract activations at the target position for each layer
            for layer_idx in layers:
                # hidden_states[layer_idx + 1] is the output of layer layer_idx
                hs = hidden_states[layer_idx + 1]  # (batch, seq, d_model)
                batch_acts = []
                for i, pos in enumerate(positions):
                    batch_acts.append(hs[i, pos].cpu())
                all_activations[layer_idx].append(torch.stack(batch_acts))

        # Concatenate across batches
        result_acts = {}
        for layer_idx in layers:
            result_acts[layer_idx] = torch.cat(all_activations[layer_idx], dim=0)

        return ExtractionResult(
            activations=result_acts,
            token_positions=all_positions,
            prompts=prompts,
        )

    @torch.no_grad()
    def extract_with_perturbation(
        self,
        prompts: list[str],
        layer: int,
        perturbation: torch.Tensor,
        token_position: str = "last",
        batch_size: int = 8,
    ) -> torch.Tensor:
        """Extract final output after adding perturbation at a specific layer.

        Used for perturbation sensitivity (λ) measurement.

        Args:
            prompts: List of prompt strings.
            layer: Layer at which to inject perturbation.
            perturbation: Tensor of shape (n_prompts, d_model) to add.
            token_position: Token position strategy.
            batch_size: Batch size.

        Returns:
            Output logits tensor of shape (n_prompts, vocab_size).
        """
        all_outputs = []
        hook_handles = []

        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start:batch_start + batch_size]
            batch_pert = perturbation[batch_start:batch_start + batch_size]

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.device)

            # With left-padding, last real token is always at the rightmost position
            seq_len = inputs["input_ids"].shape[1]
            positions = torch.full(
                (len(batch_prompts),), seq_len - 1, device=self.device
            )

            # Set up hook to inject perturbation
            batch_pert_device = batch_pert.to(self.device, dtype=self.dtype)

            def make_hook(pert, pos):
                def hook_fn(module, input, output):
                    # output is (hidden_states, ...) or just hidden_states
                    if isinstance(output, tuple):
                        hs = output[0]
                    else:
                        hs = output
                    for i, p in enumerate(pos):
                        hs[i, p] += pert[i]
                    if isinstance(output, tuple):
                        return (hs,) + output[1:]
                    return hs
                return hook_fn

            # Hook into the target layer
            target_layer = self.model.model.layers[layer]
            handle = target_layer.register_forward_hook(
                make_hook(batch_pert_device, positions)
            )
            hook_handles.append(handle)

            outputs = self.model(**inputs, return_dict=True)

            # Extract logits at target position
            logits = outputs.logits  # (batch, seq, vocab)
            batch_out = []
            for i, pos in enumerate(positions):
                batch_out.append(logits[i, pos].cpu())
            all_outputs.append(torch.stack(batch_out))

            handle.remove()

        return torch.cat(all_outputs, dim=0)

    @torch.no_grad()
    def get_baseline_output(
        self,
        prompts: list[str],
        token_position: str = "last",
        batch_size: int = 8,
    ) -> torch.Tensor:
        """Get baseline output logits without perturbation.

        Returns:
            Logits tensor of shape (n_prompts, vocab_size).
        """
        all_outputs = []

        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start:batch_start + batch_size]

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.device)

            outputs = self.model(**inputs, return_dict=True)
            logits = outputs.logits

            # With left-padding, last real token is always at the rightmost position
            batch_out = []
            for i in range(len(batch_prompts)):
                batch_out.append(logits[i, -1].cpu())
            all_outputs.append(torch.stack(batch_out))

        return torch.cat(all_outputs, dim=0)

    @torch.no_grad()
    def get_next_token_probs(
        self,
        prompts: list[str],
        batch_size: int = 8,
    ) -> torch.Tensor:
        """Get next-token probability distributions.

        Returns:
            Probability tensor of shape (n_prompts, vocab_size).
        """
        logits = self.get_baseline_output(prompts, batch_size=batch_size)
        return torch.softmax(logits.float(), dim=-1)

    @torch.no_grad()
    def evaluate_candidates(
        self,
        prompts: list[str],
        candidates_per_prompt: list[list[str]],
        batch_size: int = 8,
    ) -> tuple[list[int], list[list[float]]]:
        """Evaluate which candidate answer the model prefers for each prompt.

        Uses next-token probability — no MCQ formatting needed.

        Args:
            prompts: Prompt texts (e.g., "The capital of France is").
            candidates_per_prompt: For each prompt, list of candidate answers.
            batch_size: Batch size.

        Returns:
            (model_choices, candidate_probs):
                model_choices: Index of highest-prob candidate per prompt.
                candidate_probs: Probability of each candidate per prompt.
        """
        all_probs = self.get_next_token_probs(prompts, batch_size=batch_size)

        model_choices = []
        candidate_probs = []

        for i, candidates in enumerate(candidates_per_prompt):
            probs = []
            for candidate in candidates:
                # Try both with and without space prefix, take the higher prob
                best_prob = 0.0
                for text in [" " + candidate, candidate]:
                    token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                    if len(token_ids) == 1:
                        p = all_probs[i, token_ids[0]].item()
                        best_prob = max(best_prob, p)

                # Fallback: if neither is single-token, use first token of bare
                if best_prob == 0.0:
                    token_ids = self.tokenizer.encode(candidate, add_special_tokens=False)
                    if token_ids:
                        best_prob = all_probs[i, token_ids[0]].item()

                probs.append(best_prob)

            model_choices.append(int(np.argmax(probs)))
            candidate_probs.append(probs)

        return model_choices, candidate_probs

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 32,
        batch_size: int = 8,
    ) -> list[str]:
        """Generate completions for prompts."""
        all_completions = []

        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start:batch_start + batch_size]

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.device)

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

            # Decode only the new tokens
            for i, prompt in enumerate(batch_prompts):
                input_len = inputs["attention_mask"][i].sum().item()
                completion = self.tokenizer.decode(
                    output_ids[i, input_len:], skip_special_tokens=True
                )
                all_completions.append(completion)

        return all_completions

    @torch.no_grad()
    def generate_with_perturbation(
        self,
        prompts: list[str],
        layer: int,
        direction: np.ndarray,
        alpha: float = 1.0,
        max_new_tokens: int = 32,
        batch_size: int = 8,
    ) -> list[str]:
        """Generate completions with a steering direction added at every step.

        The direction is added to the last token position at the target layer
        on every forward pass during generation (including the prompt pass
        and each autoregressive step).

        Args:
            prompts: List of prompt strings.
            layer: Layer at which to inject the direction.
            direction: Steering direction of shape (d_model,). Added as-is
                (caller handles sign and unit normalization).
            alpha: Scalar multiplier for the direction.
            max_new_tokens: Maximum tokens to generate.
            batch_size: Batch size.

        Returns:
            List of completion strings (new tokens only).
        """
        direction_t = torch.tensor(
            direction, dtype=self.dtype, device=self.device
        ) * alpha

        all_completions = []

        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start:batch_start + batch_size]

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.device)

            # Hook: add direction at the last token position on every forward pass.
            # During prompt processing, last position = end of prompt.
            # During generation, sequence length is 1 (with KV cache), so position 0.
            def make_hook(d):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hs = output[0]
                    else:
                        hs = output
                    # Add to last position in current sequence
                    hs[:, -1] += d
                    if isinstance(output, tuple):
                        return (hs,) + output[1:]
                    return hs
                return hook_fn

            target_layer = self.model.model.layers[layer]
            handle = target_layer.register_forward_hook(make_hook(direction_t))

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

            handle.remove()

            for i, prompt in enumerate(batch_prompts):
                input_len = inputs["attention_mask"][i].sum().item()
                completion = self.tokenizer.decode(
                    output_ids[i, input_len:], skip_special_tokens=True
                )
                all_completions.append(completion)

        return all_completions
