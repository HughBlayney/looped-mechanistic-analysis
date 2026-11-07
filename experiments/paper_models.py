"""Ordered model groups shared by the paper's experiment notebooks."""

MAIN_PARCAE_MODEL = "parcae_770m"

PAPER_MODEL_GROUPS = {
    "main": (
        "ouro_1_4b",
        "raven",
        "retrofitted_llama",
        MAIN_PARCAE_MODEL,
    ),
    "appendix": (
        "retrofitted_olmo",
        "retrofitted_tinyllama",
        "parcae_140m",
        "parcae_370m",
    ),
}
# Preserve main/appendix panel order in combined figures.
PAPER_MODEL_GROUPS["all"] = PAPER_MODEL_GROUPS["main"] + PAPER_MODEL_GROUPS["appendix"]


def paper_model_pairs(recurrences, model_recurrences=None):
    """Resolve unique collection pairs and ordered main/appendix/all panels."""
    if model_recurrences is None:
        model_recurrences = list(
            dict.fromkeys((name, recurrences) for name in PAPER_MODEL_GROUPS["all"])
        )
    pairs = [tuple(pair) for pair in model_recurrences]
    if not pairs or len(set(pairs)) != len(pairs):
        raise ValueError(
            "Select at least one model-recurrence pair, without duplicates"
        )
    if unknown := {name for name, _ in pairs} - set(PAPER_MODEL_GROUPS["all"]):
        raise ValueError(f"Models outside the shared paper groups: {sorted(unknown)}")
    figure_pairs = {
        group: [pair for name in names for pair in pairs if pair[0] == name]
        for group, names in PAPER_MODEL_GROUPS.items()
    }
    return pairs, figure_pairs
