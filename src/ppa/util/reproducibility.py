"""Global reproducibility utilities: seeds and deterministic settings."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility.

    Args:
        seed: Integer seed value to use globally.

    Note:
        PYTHONHASHSEED should ideally be set before the interpreter starts.
        Setting it here will warn if the interpreter was not started with it.
    """
    current_hash_seed = os.environ.get("PYTHONHASHSEED")
    if current_hash_seed is None or current_hash_seed != str(seed):
        os.environ["PYTHONHASHSEED"] = str(seed)
        logger.warning(
            "PYTHONHASHSEED set to %d at runtime; "
            "for full reproducibility, set it before starting Python.",
            seed,
        )

    random.seed(seed)
    np.random.seed(seed)
    logger.info("Global seed set to %d", seed)
