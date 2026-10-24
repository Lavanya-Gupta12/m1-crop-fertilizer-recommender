"""Task 2: which fertilizer source answers for each stage-1 crop.

Tier 1 - the crop itself is a label in the fertilizer dataset -> ML model with that crop.
Tier 2 - crop-group proxy: the crop belongs to a group that the fertilizer dataset labels as a
         single class -> ML model queried with the group label. See docs/fertilizer_tiers.md
         for the assumption and its limits.
None   - no verified fertilizer data; no recommendation is made.
"""

TIER1 = {"rice", "maize", "cotton"}

# stage-1 crop -> fertilizer-dataset group label
TIER2_PROXY = {
    "chickpea": "pulses", "kidneybeans": "pulses", "pigeonpeas": "pulses", "mothbeans": "pulses",
    "mungbean": "pulses", "blackgram": "pulses", "lentil": "pulses",
}

NO_DATA_NOTE = "No verified fertilizer data for this crop."


def resolve(crop):
    """Return (tier, fertilizer-dataset crop label or None, human-readable provenance)."""
    if crop in TIER1:
        return 1, crop, f"Tier 1: ML model trained on '{crop}' rows."
    if crop in TIER2_PROXY:
        group = TIER2_PROXY[crop]
        return 2, group, (f"Tier 2: proxy via crop group - '{crop}' answered with the "
                          f"'{group}' rows of the fertilizer dataset (assumption, not crop-specific data).")
    return None, None, NO_DATA_NOTE


def coverage(stage1_crops):
    tiers = {str(c): resolve(str(c))[0] for c in stage1_crops}
    return {"tier1": sorted(c for c, t in tiers.items() if t == 1),
            "tier2": sorted(c for c, t in tiers.items() if t == 2),
            "none": sorted(c for c, t in tiers.items() if t is None)}
