"""Loader for concept family datasets."""

from src.data.base import ProbePrompt
from src.data.core_families import (
    ArithmeticProbe,
    GeographyProbe,
    SequenceProbe,
    WordTransformProbe,
    AnalogyProbe,
)
from src.data.controlled_families import (
    ParityProbe,
    ContinentBinaryProbe,
    AnimalBinaryProbe,
    GenderProbe,
    TemperatureProbe,
    SizeProbe,
    SpeedProbe,
    MaterialProbe,
    HabitatProbe,
    IndoorOutdoorProbe,
    EdibleProbe,
    NaturalProbe,
    PhaseProbe,
    PlantAnimalProbe,
    PluralProbe,
    SentimentProbe,
    AgeProbe,
    DangerProbe,
    VolumeProbe,
    WeightProbe,
    RoundFlatProbe,
    HardSoftProbe,
    WetDryProbe,
    DayNightProbe,
    SweetSourProbe,
)

# Original 5 families (used in within-concept layer analysis)
CORE_FAMILIES = {
    "arithmetic": ArithmeticProbe,
    "geography": GeographyProbe,
    "sequence": SequenceProbe,
    "word_transform": WordTransformProbe,
    "analogy": AnalogyProbe,
}

# 25 controlled binary families (used for steerability prediction)
CONTROLLED_FAMILIES = {
    "c_parity": ParityProbe,
    "c_continent": ContinentBinaryProbe,
    "c_animal": AnimalBinaryProbe,
    "c_gender": GenderProbe,
    "c_temperature": TemperatureProbe,
    "c_size": SizeProbe,
    "c_speed": SpeedProbe,
    "c_material": MaterialProbe,
    "c_habitat": HabitatProbe,
    "c_indoor": IndoorOutdoorProbe,
    "c_edible": EdibleProbe,
    "c_natural": NaturalProbe,
    "c_phase": PhaseProbe,
    "c_plant": PlantAnimalProbe,
    "c_plural": PluralProbe,
    "c_sentiment": SentimentProbe,
    "c_age": AgeProbe,
    "c_danger": DangerProbe,
    "c_volume": VolumeProbe,
    "c_weight": WeightProbe,
    "c_shape": RoundFlatProbe,
    "c_hardness": HardSoftProbe,
    "c_moisture": WetDryProbe,
    "c_daynight": DayNightProbe,
    "c_taste": SweetSourProbe,
}


def load_families(
    n_prompts: int = 500,
    include_controlled: bool = False,
) -> dict[str, list[ProbePrompt]]:
    """Load concept families as next-token prediction probes.

    Args:
        n_prompts: Max prompts per family.
        include_controlled: If True, include the 25 controlled binary families.

    Returns:
        Dict mapping family name to list of ProbePrompts.
    """
    family_classes = dict(CORE_FAMILIES)
    if include_controlled:
        family_classes.update(CONTROLLED_FAMILIES)

    result = {}
    for name, cls in family_classes.items():
        dataset = cls(n_prompts=n_prompts)
        result[name] = dataset.load_validated()

    return result
