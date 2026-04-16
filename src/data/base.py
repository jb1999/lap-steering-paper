"""Base classes for concept datasets."""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class ProbePrompt:
    """A prompt for concept probing via next-token prediction.

    The model sees only the prompt_text. We check which of the
    candidate answers gets the highest next-token probability.
    No MCQ formatting — pure next-token prediction.

    Attributes:
        prompt_text: The text to feed to the model (e.g., "The capital of France is")
        candidates: List of possible next-token answers (e.g., ["Paris", "Berlin", "London"])
        correct_index: Index of the correct candidate (0-based).
        family: Which concept family.
        metadata: Additional info.
    """
    prompt_text: str
    candidates: list[str]
    correct_index: int
    family: str
    metadata: dict = field(default_factory=dict)

    @property
    def correct_answer(self) -> str:
        return self.candidates[self.correct_index]

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)


class ProbeDataset(ABC):
    """Abstract base for probe concept family datasets."""

    family_name: str = ""

    @abstractmethod
    def load(self) -> list[ProbePrompt]:
        """Load and return all probe prompts."""
        ...

    def load_validated(self) -> list[ProbePrompt]:
        """Load and validate prompts."""
        prompts = self.load()
        valid = [p for p in prompts if self._validate(p)]
        print(f"[{self.family_name}] Loaded {len(valid)}/{len(prompts)} valid prompts")
        return valid

    def _validate(self, p: ProbePrompt) -> bool:
        return (
            bool(p.prompt_text.strip())
            and len(p.candidates) >= 1
            and 0 <= p.correct_index < len(p.candidates)
            and len(set(p.candidates)) == len(p.candidates)  # no duplicates
        )
