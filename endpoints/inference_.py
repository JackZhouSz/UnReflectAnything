"""Inference API for UnReflectAnything."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from os import PathLike

if False:
    from torch import Tensor


def inference(
    input: Union[str, PathLike, Path, "Tensor"],
    output: Optional[Union[str, PathLike, Path]] = None,
    weights_path: Optional[Union[str, PathLike, Path]] = None,
    config: Optional[Union[str, PathLike, Path, dict]] = None,
    device: str = "cuda",
    batch_size: int = 4,
    brightness_threshold: float = 0.8,
    resize_output: bool = True,
    verbose: bool = False,
) -> Optional["Tensor"]:
    """Run inference on input image(s) to remove specular reflections.

    This function runs the UnReflectAnything model on input images to produce
    diffuse (reflection-free) outputs. It supports both file-based and tensor-based
    workflows.

    Args:
        input: Input source. Can be:
            - Path to a single image file
            - Path to a directory containing images
            - Tensor of shape [B, 3, H, W] (batch of RGB images, values in [0, 1])
        output: Output destination. If provided, results are saved to disk.
            If None, returns the result as a tensor.
        weights_path: Path to model weights. Defaults to the cache directory.
        config: Configuration source (YAML path or dict). If None, uses default.
        device: Device to run inference on (e.g. 'cuda', 'cpu').
        batch_size: Number of images to process per forward pass (default: 4).
        brightness_threshold: Threshold for highlight mask computation (0.0-1.0).
        resize_output: If True, resize output images to match original input dimensions.
        verbose: If True, print progress information.

    Returns:
        If output is None: Tensor of shape [B, 3, H, W] with diffuse predictions.
        If output is provided: None (results saved to disk).
    """
    from torch import Tensor

    from inference import InferenceOptions, run_inference as _run_inference_files

    from unreflectanything._shared import (
        DEFAULT_WEIGHTS_FILENAME,
        get_cache_dir,
        _resolve_device,
        _apply_config_to_options,
    )

    is_tensor_input = isinstance(input, Tensor)

    if is_tensor_input:
        return _inference_tensor(
            input_tensor=input,
            weights_path=weights_path,
            config=config,
            device=device,
            brightness_threshold=brightness_threshold,
            verbose=verbose,
        )

    input_path = Path(input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if output is None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            return _inference_files_return_tensors(
                input_path=input_path,
                output_path=output_path,
                weights_path=weights_path,
                config=config,
                device=device,
                batch_size=batch_size,
                brightness_threshold=brightness_threshold,
                resize_output=resize_output,
                verbose=verbose,
            )

    output_path = Path(output).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if weights_path is None:
        resolved_weights = get_cache_dir("weights") / DEFAULT_WEIGHTS_FILENAME
    else:
        resolved_weights = Path(weights_path).expanduser().resolve()

    if not resolved_weights.exists():
        raise FileNotFoundError(
            f"Weights not found at {resolved_weights}. "
            "Run 'unreflect download --weights' first."
        )

    options = InferenceOptions(
        weights_path=resolved_weights,
        input_dir=input_path if input_path.is_dir() else input_path.parent,
        output_dir=output_path if output_path.is_dir() else output_path.parent,
        device=_resolve_device(device),
        batch_size=batch_size,
        brightness_threshold=brightness_threshold,
        resize_output=resize_output,
        monitor_usage=False,
    )

    if config is not None:
        options = _apply_config_to_options(options, config)

    _run_inference_files(options)
    return None


def _inference_tensor(
    input_tensor: "Tensor",
    weights_path: Optional[Union[str, Path]] = None,
    config: Optional[Union[str, Path, dict]] = None,
    device: str = "cuda",
    brightness_threshold: float = 0.8,
    verbose: bool = False,
) -> "Tensor":
    """Run inference on a tensor input, returning a tensor output."""
    import torch
    from inference import InferenceOptions, load_model

    from unreflectanything._shared import (
        DEFAULT_WEIGHTS_FILENAME,
        get_cache_dir,
        _resolve_device,
        _apply_config_to_options,
    )

    if input_tensor.dim() != 4:
        raise ValueError(f"Input tensor must be 4D [B,C,H,W], got {input_tensor.dim()}D")
    if input_tensor.shape[1] != 3:
        raise ValueError(f"Input tensor must have 3 channels, got {input_tensor.shape[1]}")

    if weights_path is None:
        resolved_weights = get_cache_dir("weights") / DEFAULT_WEIGHTS_FILENAME
    else:
        resolved_weights = Path(weights_path).expanduser().resolve()

    if not resolved_weights.exists():
        raise FileNotFoundError(
            f"Weights not found at {resolved_weights}. "
            "Run 'unreflect download --weights' first."
        )

    options = InferenceOptions(
        weights_path=resolved_weights,
        input_dir=Path("."),
        output_dir=Path("."),
        device=device,
        brightness_threshold=brightness_threshold,
    )

    if config is not None:
        options = _apply_config_to_options(options, config)

    torch_device = torch.device(_resolve_device(device))
    model = load_model(options, torch_device)

    input_tensor = input_tensor.to(device=torch_device, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        outputs = model({"rgb": input_tensor})

    diffuse = outputs.get("diffuse")
    if diffuse is None:
        raise KeyError("Model output does not contain 'diffuse'")

    return diffuse.clamp(0.0, 1.0)


def _inference_files_return_tensors(
    input_path: Path,
    output_path: Path,
    weights_path: Optional[Union[str, Path]],
    config: Optional[Union[str, Path, dict]],
    device: str,
    batch_size: int,
    brightness_threshold: float,
    resize_output: bool,
    verbose: bool,
) -> "Tensor":
    """Run file-based inference but return results as tensors instead of saving."""
    import torch
    from PIL import Image
    from torchvision.transforms import functional as TF

    from inference import InferenceOptions, list_image_paths, load_model

    from unreflectanything._shared import (
        DEFAULT_WEIGHTS_FILENAME,
        get_cache_dir,
        _resolve_device,
        _apply_config_to_options,
    )

    if weights_path is None:
        resolved_weights = get_cache_dir("weights") / DEFAULT_WEIGHTS_FILENAME
    else:
        resolved_weights = Path(weights_path).expanduser().resolve()

    if not resolved_weights.exists():
        raise FileNotFoundError(
            f"Weights not found at {resolved_weights}. "
            "Run 'unreflect download --weights' first."
        )

    options = InferenceOptions(
        weights_path=resolved_weights,
        input_dir=input_path if input_path.is_dir() else input_path.parent,
        output_dir=output_path,
        device=device,
        batch_size=batch_size,
        brightness_threshold=brightness_threshold,
        resize_output=resize_output,
    )

    if config is not None:
        options = _apply_config_to_options(options, config)

    torch_device = torch.device(_resolve_device(device))
    model = load_model(options, torch_device)
    target_side = model.dinov3.config["image_size"]
    target_size = (target_side, target_side)

    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = list_image_paths(input_path, options.image_extensions)

    results = []
    model.eval()

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_tensors = []
        for path in batch_paths:
            with Image.open(path) as img:
                rgb_img = img.convert("RGB")
                tensor = TF.to_tensor(rgb_img)
                resized = TF.resize(tensor, target_size, antialias=True)
                batch_tensors.append(resized)

        rgb_batch = torch.stack(batch_tensors, dim=0).to(
            device=torch_device, dtype=torch.float32
        )

        with torch.no_grad():
            outputs = model({"rgb": rgb_batch})

        diffuse = outputs.get("diffuse")
        if diffuse is None:
            raise KeyError("Model output does not contain 'diffuse'")

        results.append(diffuse.clamp(0.0, 1.0).cpu())

    return torch.cat(results, dim=0)
