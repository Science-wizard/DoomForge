"""Single source of truth: the model lives in submission/model.py (the artifact
that is evaluated).  Training imports it from there so the two can never drift."""
from submission.model import (  # noqa: F401
    HIDDEN, HEADS, HEAD_SIZES, MAX_ENTITIES, NUM_ENTITY_TYPES, NUM_RAYS, NUM_RAY_TYPES, VITALS,
    DoomNet, ModelConfig, actions_to_vector, vector_to_actions,
)
