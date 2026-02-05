"""UnReflectAnything: deep learning method for removing specular reflections from RGB images.

Public API:
    run_pipeline: Train or test the network (main entry for training/testing).
    run_inference: Run inference on an image directory from InferenceOptions.
    InferenceOptions: Dataclass for inference configuration.
    compute_highlight_mask: Compute binary highlight masks from RGB batch.
    get_weights_cache_dir: Default cache directory for downloaded weights.
"""

from __future__ import annotations

# Re-export main pipeline entry point
import main as _main

run_pipeline = _main.run_pipeline

# Re-export inference API
from inference import (
    InferenceOptions,
    compute_highlight_mask,
    run_inference,
)

# Optional: high-level helper for running inference from paths (uses default/cache weights)
from unreflectanything.weights import get_weights_cache_dir

__all__ = [
    "run_pipeline",
    "run_inference",
    "InferenceOptions",
    "compute_highlight_mask",
    "get_weights_cache_dir",
]
