"""Python API for UnReflectAnything.

This module provides the core API functions that both the CLI and programmatic
users call. The CLI is a thin wrapper around these functions.

Example usage:
    from unreflectanything import inference, evaluate

    # File-based inference (saves to disk)
    inference("input.png", output="output.png")

    # Tensor-based inference (returns tensor)
    import torch
    img = torch.randn(1, 3, 448, 448)  # [B, C, H, W]
    result = inference(img)  # Returns [B, 3, H, W] tensor

    # Evaluate results
    metrics = evaluate("output/", "reference/", metrics=["psnr", "ssim"])
"""


import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Sequence, Tuple, Union
from os import PathLike

try:
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision.transforms import functional as TF
except ImportError:
    Dataset = None  # type: ignore[misc, assignment]
    Image = None  # type: ignore[misc, assignment]
    TF = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from os import PathLike
    from torch import Tensor

# Used only as base for UnReflectModel; imported lazily in class definition
def _nn_module_base():
    import torch.nn as nn
    return nn.Module


# =============================================================================
# IMAGE DATASET
# =============================================================================

# Default extensions consistent with inference.py
DEFAULT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _collect_image_paths(
    root: Path,
    extensions: Sequence[str],
) -> List[Path]:
    """Collect image paths under root matching extensions (case-insensitive)."""
    lower_exts = tuple(ext.lower() for ext in extensions)
    paths = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in lower_exts
    ]
    return sorted(paths)


if Dataset is not None:
    class ImageDirDataset(Dataset):  # type: ignore[no-redef]
        """
        Dataset that reads images from a directory and returns tensors.

        Each item is a tensor of shape (3, H, W) in [0, 1], optionally resized.
        """

        def __init__(
            self,
            root_dir: Union[str, PathLike, Path],
            extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
            target_size: Optional[Tuple[int, int]] = None,
            return_path: bool = False,
        ):
            """
            Args:
                root_dir: Directory to scan for images (recursive).
                extensions: File suffixes to consider (e.g. (".png", ".jpg")).
                target_size: If set, (H, W) to resize each image; antialias used.
                return_path: If True, __getitem__ returns (tensor, path_str).
            """
            self.root = Path(root_dir)
            self.extensions = tuple(extensions)
            self.target_size = target_size  # (H, W) or None
            self.return_path = return_path
            self.paths = _collect_image_paths(self.root, self.extensions)
            if not self.paths:
                raise FileNotFoundError(f"No images found under {self.root}")

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, idx: int):
            path = self.paths[idx]
            with Image.open(path) as img:
                rgb = img.convert("RGB")
                x = TF.to_tensor(rgb)  # (3, H, W), float32, [0, 1]
            if self.target_size is not None:
                x = TF.resize(x, self.target_size, antialias=True)  # (3, H_t, W_t)
            if self.return_path:
                return x, str(path)
            return x
else:
    ImageDirDataset = None  # type: ignore[misc, assignment]


# =============================================================================
# INFERENCE
# =============================================================================

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
        output: Output destination. If provided, results are saved to disk:
            - Path to output file (for single image input)
            - Path to output directory (for directory input)
            If None, returns the result as a tensor.
        weights_path: Path to model weights. Defaults to the cache directory
            (~/.cache/unreflectanything/weights/full_model_weights.pt).
        config: Configuration source. Can be:
            - Path to a YAML config file
            - Dictionary with config overrides
            If None, uses default config_inference.yaml settings.
        device: Device to run inference on. Use ``'cuda'`` (or ``'cuda:0'`` when
            multiple GPUs exist), ``'cuda:0'``, ``'cuda:1'``, etc., or ``'cpu'``.
            Default ``'cuda'`` uses the single GPU when only one is available.
        batch_size: Number of images to process per forward pass (default: 4).
        brightness_threshold: Threshold for highlight mask computation (0.0-1.0).
            Pixels with brightness above this value are considered highlights.
        resize_output: If True, resize output images to match original input
            dimensions. Only applies when saving to files.
        verbose: If True, print progress information.

    Returns:
        If output is None: Tensor of shape [B, 3, H, W] with diffuse predictions.
        If output is provided: None (results saved to disk).

    Raises:
        FileNotFoundError: If input path or weights_path doesn't exist.
        ValueError: If input tensor has invalid shape.

    Example:
        >>> # File-based inference
        >>> inference("input.png", output="output.png")
        
        >>> # Tensor-based inference
        >>> img = torch.rand(1, 3, 448, 448)
        >>> result = inference(img)  # Returns [1, 3, 448, 448] tensor
    """
    # Import here to avoid circular imports and speed up module load
    
    from torch import Tensor

    from inference import (
        InferenceOptions,
    )
    from inference import (
        run_inference as _run_inference_files,
    )
    from unreflectanything.weights import (
        DEFAULT_WEIGHTS_FILENAME,
        get_cache_dir,
    )

    # Determine if input is a tensor
    is_tensor_input = isinstance(input, Tensor)

    if is_tensor_input:
        # Tensor-based inference path
        return _inference_tensor(
            input_tensor=input,
            weights_path=weights_path,
            config=config,
            device=device,
            brightness_threshold=brightness_threshold,
            verbose=verbose,
        )
    else:
        # File-based inference path
        input_path = Path(input).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")

        # Determine output path
        if output is None:
            # Create a temporary output directory and return tensors
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
        else:
            output_path = Path(output).expanduser().resolve()
            output_path.mkdir(parents=True, exist_ok=True)

            # Resolve weights path
            if weights_path is None:
                resolved_weights = get_cache_dir("weights") / DEFAULT_WEIGHTS_FILENAME
            else:
                resolved_weights = Path(weights_path).expanduser().resolve()

            if not resolved_weights.exists():
                raise FileNotFoundError(
                    f"Weights not found at {resolved_weights}. "
                    "Run 'unreflect download --weights' first."
                )

            # Build options and run file-based inference
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

            # Handle config overrides
            if config is not None:
                options = _apply_config_to_options(options, config)

            _run_inference_files(options)
            return None


# =============================================================================
# MODEL FACTORY AND WRAPPER (ura.model() / ura.model(pretrained=True))
# =============================================================================


def model(
    pretrained: bool = False,
    weights_path: Optional[Union[str, PathLike, Path]] = None,
    device: str = "cuda",
    config_path: Optional[Union[str, PathLike, Path, dict]] = None,
    verbose: bool = False,
):
    """Return the model class or a pretrained model instance callable with batched RGB.

    Use this for a lightweight API: get a callable module and run it on tensors.

    Args:
        pretrained: If False (default), return the underlying model class
            (UnReflect_Model_TokenInpainter) for custom instantiation or training.
            If True, return an ``UnReflectModel`` instance with weights loaded,
            which you can call with a batched RGB tensor.
        weights_path: Path to checkpoint. Only used when pretrained=True.
            Defaults to cache (~/.cache/unreflectanything/weights/full_model_weights.pt).
        device: Device to load the model on when pretrained=True (e.g. ``"cuda"``, ``"cpu"``).
        config_path: Optional config source (YAML path or dict) for architecture when
            loading from checkpoint. Only used when pretrained=True.

    Returns:
        If pretrained=False: the model class (UnReflect_Model_TokenInpainter).
        If pretrained=True: an ``UnReflectModel`` instance (nn.Module) that can
        be called with ``model(images)`` where images is [B, 3, H, W].

    Example:
        >>> import unreflectanything as ura
        >>> uramodel = ura.model(pretrained=True)
        >>> images = torch.rand(2, 3, 448, 448, device="cuda")  # [B, 3, H, W]
        >>> diffuse = uramodel(images)  # [B, 3, H, W]
        >>> # Or get the class only (e.g. for training)
        >>> ModelClass = ura.model()
    """
    
    from main import create_model_from_config, load_and_process_config   
    from pathlib import Path
    from unreflectanything.weights import get_cache_dir
    
    if config_path is None:
        config_path = get_cache_dir("weights").parent / "configs" / "pretrained_config.yaml"
    if config_path is not None and config_path.is_dir():
        config_path = Path(config_path)
        config_path = config_path / "pretrained_config.yaml"
    model_config = load_and_process_config(config_path)
    if verbose:
        print(f"Loaded model configuration from: `{config_path}`")
        
    if not pretrained:
        return create_model_from_config(model_config, device, verbose=verbose)
    # Check that weights_path exists and contains "full_model.pth"
    resolved_weights = None
    if weights_path is not None:
        
        resolved_weights = Path(weights_path).expanduser().resolve()
        if not resolved_weights.exists():
            raise FileNotFoundError(f"Weights file not found at '{resolved_weights}'.\n Please run 'unreflect download --weights' or 'unreflectanything.download('weights') first.")
        if "full_model_weights.pth" not in resolved_weights.name:
            raise ValueError(
                f"Cannot find full model weights in '{resolved_weights}.\n Please run 'unreflect download --weights' or 'unreflectanything.download('weights') first."
            )
        
    return UnReflectModel(
        pretrained=True,
        weights_path=weights_path,
        device=device,
        config=model_config,
        verbose=verbose,
    )


class UnReflectModel(_nn_module_base()):
    """Thin wrapper (nn.Module) around the loaded UnReflect model for tensor-in, tensor-out inference.

    This wrapper is callable so that ``model(images)`` returns the diffuse
    prediction tensor. Use ``ura.model(pretrained=True)`` to obtain an instance.
    The inner model is stored as a submodule so ``.to(device)``, ``.eval()``,
    and ``.parameters()`` work as expected.

    Attributes:
        image_size: Expected spatial size (side) for the inner encoder (e.g. 448).
        device: Device the  model lives on (read-only).
    """

    def __init__(
        self,
        pretrained: bool = True,
        weights_path: Optional[Union[str, PathLike, Path]] = None,
        device: str = "cuda",
        config: Optional[Union[str, PathLike, Path, dict]] = None,
        verbose: bool = False,
    ):
        if not pretrained:
            raise ValueError("UnReflectModel(pretrained=False) is not supported; use ura.model() to get the class.")
        super().__init__()
        from inference import InferenceOptions, load_model
        from unreflectanything.weights import (
            DEFAULT_WEIGHTS_FILENAME,
            get_cache_dir,
        )

        if weights_path is None:
            resolved_weights = get_cache_dir("weights") / DEFAULT_WEIGHTS_FILENAME
        else:
            resolved_weights = Path(weights_path).expanduser().resolve()
        if not resolved_weights.exists():
            raise FileNotFoundError(
                f"Weights not found at {resolved_weights}. Run 'unreflect download --weights' first."
            )

        options = InferenceOptions(
            weights_path=resolved_weights,
            input_dir=Path("."),
            output_dir=Path("."),
            device=device,
        )
        if config is not None:
            options = _apply_config_to_options(options, config)

        torch_device = __import__("torch").device(_resolve_device(device))
        inner = load_model(options, torch_device, verbose=verbose)
        self._model = inner  # registered as submodule by nn.Module
        self._device = torch_device
        cfg = self._model.dinov3.config
        self.image_size = getattr(cfg, "image_size", cfg.get("image_size", 448) if hasattr(cfg, "get") else 448)

    @property
    def device(self):
        return self._device

    def forward(
        self,
        images: "Tensor",
        inpaint_mask_override: Optional["Tensor"] = None,
        return_dict: bool = False,
    ) -> Union["Tensor", Dict[str, "Tensor"]]:
        """Run inference on a batch of RGB images.

        Args:
            images: Batched RGB tensor [B, 3, H, W], values in [0, 1]. Will be
                moved to the model device and resized internally if needed by the
                encoder (see ``image_size``).
            inpaint_mask_override: Optional [B, 1, H, W] mask to force inpainting
                regions (1 = inpaint). If None, the model uses its highlight head.
            return_dict: If True, return the full output dict (e.g. ``diffuse``,
                ``highlight``, ``patch_mask``). If False, return only the
                diffuse tensor [B, 3, H, W].

        Returns:
            If return_dict=False: diffuse tensor [B, 3, H, W].
            If return_dict=True: dict with at least ``diffuse``, ``highlight``, etc.
        """
        import torch

        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(f"images must be [B, 3, H, W], got shape {tuple(images.shape)}")
        batch = {
            "rgb": images.to(device=self._device, dtype=torch.float32),
        }
        if inpaint_mask_override is not None:
            batch["inpaint_mask_override"] = inpaint_mask_override.to(
                device=self._device, dtype=torch.float32
            )
        self._model.eval()
        with torch.no_grad():
            out = self._model(batch)
        diffuse = out.get("diffuse")
        if diffuse is None:
            raise KeyError("Model output does not contain 'diffuse'")
        diffuse = diffuse.clamp(0.0, 1.0)
        if return_dict:
            out["diffuse"] = diffuse
            return out
        return diffuse

    def eval(self):
        """Set the inner model to eval mode."""
        self._model.eval()
        return self

    def train(self, mode: bool = True):
        """Set the inner model to train mode (for fine-tuning)."""
        self._model.train(mode)
        return self


def _resolve_device(device: str) -> str:
    """Resolve device string for inference: use CUDA when available, else CPU.

    When device is 'cuda' and exactly one GPU is available, returns 'cuda'.
    When device is 'cuda' and multiple GPUs exist, returns 'cuda:0'.
    Otherwise returns the given device string (e.g. 'cuda:1', 'cpu').
    """
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    if device == "cuda":
        return "cuda:0" if torch.cuda.device_count() > 1 else "cuda"
    return device


def _inference_tensor(
    input_tensor: "Tensor",
    weights_path: Optional[Union[str, Path]] = None,
    config: Optional[Union[str, Path, dict]] = None,
    device: str = "cuda",
    brightness_threshold: float = 0.8,
    verbose: bool = False,
) -> "Tensor":
    """Run inference on a tensor input, returning a tensor output.

    This is the minimal-overhead inference path for programmatic use.

    Args:
        input_tensor: Input tensor of shape [B, 3, H, W], values in [0, 1].
        weights_path: Path to model weights.
        config: Configuration source.
        device: Device to run on.
        brightness_threshold: Highlight threshold.
        verbose: Print progress.

    Returns:
        Tensor of shape [B, 3, H, W] with diffuse predictions.
    """
    import torch
    from inference import InferenceOptions, load_model
    from unreflectanything.weights import (
        DEFAULT_WEIGHTS_FILENAME,
        get_cache_dir,
    )

    # Validate input tensor
    if input_tensor.dim() != 4:
        raise ValueError(f"Input tensor must be 4D [B,C,H,W], got {input_tensor.dim()}D")
    if input_tensor.shape[1] != 3:
        raise ValueError(f"Input tensor must have 3 channels, got {input_tensor.shape[1]}")

    # Resolve weights path
    if weights_path is None:
        resolved_weights = get_cache_dir("weights") / DEFAULT_WEIGHTS_FILENAME
    else:
        resolved_weights = Path(weights_path).expanduser().resolve()

    if not resolved_weights.exists():
        raise FileNotFoundError(
            f"Weights not found at {resolved_weights}. "
            "Run 'unreflect download --weights' first."
        )

    # Create minimal options for model loading
    options = InferenceOptions(
        weights_path=resolved_weights,
        input_dir=Path("."),  # Placeholder, not used for tensor inference
        output_dir=Path("."),  # Placeholder, not used for tensor inference
        device=device,
        brightness_threshold=brightness_threshold,
    )

    if config is not None:
        options = _apply_config_to_options(options, config)

    # Load model
    torch_device = torch.device(_resolve_device(device))
    model = load_model(options, torch_device)

    # Move input to device
    input_tensor = input_tensor.to(device=torch_device, dtype=torch.float32)

    # Compute highlight mask
    # inpaint_mask = compute_highlight_mask(input_tensor, threshold=brightness_threshold)

    # Run inference - minimal forward pass
    model.eval()
    with torch.no_grad():
        outputs = model({
            "rgb": input_tensor,
        })

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

    from inference import (
        InferenceOptions,
        list_image_paths,
        load_model,
    )
    from unreflectanything.weights import (
        DEFAULT_WEIGHTS_FILENAME,
        get_cache_dir,
    )

    # Resolve weights path
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

    # Collect image paths
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = list_image_paths(input_path, options.image_extensions)

    # Process all images and collect results
    results = []
    model.eval()

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        
        # Load and preprocess batch
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
            outputs = model({
                "rgb": rgb_batch,
            })

        diffuse = outputs.get("diffuse")
        if diffuse is None:
            raise KeyError("Model output does not contain 'diffuse'")

        results.append(diffuse.clamp(0.0, 1.0).cpu())

    # Concatenate all results
    return torch.cat(results, dim=0)


def _apply_config_to_options(options, config):
    """Apply config overrides to inference options."""
    from typing import TYPE_CHECKING

    import yaml

    if TYPE_CHECKING:
        pass

    if isinstance(config, (str, Path)):
        config_path = Path(config).expanduser().resolve()
        if config_path.exists():
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            config_dict = {}
    elif isinstance(config, dict):
        config_dict = config
    else:
        return options

    # Apply overrides
    if "batch_size" in config_dict:
        options.batch_size = int(config_dict["batch_size"])
    if "device" in config_dict:
        options.device = config_dict["device"]
    if "brightness_threshold" in config_dict:
        options.brightness_threshold = float(config_dict["brightness_threshold"])
    if "resize_output" in config_dict:
        options.resize_output = bool(config_dict["resize_output"])
    if "num_workers" in config_dict:
        options.num_workers = int(config_dict["num_workers"])

    return options


# =============================================================================
# TRAINING
# =============================================================================

def train(
    config: Union[str, PathLike, Path] = "config_train.yaml",
    resume_run: Optional[str] = None,
    boot: bool = False,
    **overrides,
) -> None:
    """Run the training pipeline.

    This function trains the UnReflectAnything model using the specified
    configuration. It supports resuming from checkpoints and overriding
    config parameters via keyword arguments.

    Args:
        config: Path to the training configuration YAML file.
        resume_run: Run identifier to resume training from. If provided,
            training continues from the last checkpoint of the specified run.
        boot: If True, run in boot mode with minimal parameters for quick testing
            (batch_size=1, epochs=1, no_wandb=True).
        **overrides: Additional config overrides in the format PARAM=value.
            These override values from the config file.

    Returns:
        None

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        RuntimeError: If training fails.

    Example:
        >>> # Train with default config
        >>> train()
        
        >>> # Train with custom config and overrides
        >>> train("my_config.yaml", EPOCHS=50, BATCH_SIZE=32)
        
        >>> # Resume training from a previous run
        >>> train(resume_run="gallant-bush-806")
    """
    # Build argv for the training pipeline
    config_path = Path(config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Prepare sys.argv for main.run_pipeline
    argv_backup = sys.argv
    new_argv = [sys.argv[0], "--config", str(config_path)]

    if resume_run:
        new_argv.extend(["--resume-run", resume_run])
    if boot:
        new_argv.append("--boot")

    # Add overrides
    for key, value in overrides.items():
        new_argv.append(f"--{key.upper()}={value}")

    try:
        sys.argv = new_argv
        import main
        main.run_pipeline(mode="train")
    finally:
        sys.argv = argv_backup


# =============================================================================
# TESTING
# =============================================================================

def test(
    config: Union[str, PathLike, Path] = "config_test.yaml",
    **overrides,
) -> None:
    """Run the test/evaluation pipeline.

    This function evaluates a trained UnReflectAnything model using the
    specified configuration. The model checkpoint is determined by the
    RUN parameter in the config.

    Args:
        config: Path to the test configuration YAML file.
        **overrides: Additional config overrides in the format PARAM=value.
            These override values from the config file.

    Returns:
        None

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        RuntimeError: If testing fails or RUN is not specified.

    Example:
        >>> # Test with default config
        >>> test()
        
        >>> # Test a specific run
        >>> test(RUN="gallant-bush-806")
    """
    config_path = Path(config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Prepare sys.argv for main.run_pipeline
    argv_backup = sys.argv
    new_argv = [sys.argv[0], "--config", str(config_path)]

    # Add overrides
    for key, value in overrides.items():
        new_argv.append(f"--{key.upper()}={value}")

    try:
        sys.argv = new_argv
        import main
        main.run_pipeline(mode="test")
    finally:
        sys.argv = argv_backup


# =============================================================================
# DOWNLOAD
# =============================================================================

def download(
    what: Optional[Literal["weights", "images", "notebooks", "configs", "all"]] = None,
    *,
    asset: Optional[Literal["weights", "images", "notebooks", "configs", "all"]] = None,
    output_dir: Optional[Union[str, PathLike, Path]] = None,
    variant: str = "default",
    force: bool = False,
) -> Path:
    """Download assets from the HuggingFace repository.

    This function downloads pretrained weights, sample images, or example
    notebooks from the UnReflectAnything HuggingFace repository. Same
    behavior as the CLI ``unreflect download --weights`` / ``--images`` /
    ``--notebooks`` / ``--all``.

    Args:
        what: What to download (positional). Use ``asset=`` as alias.
            - "weights": Pretrained model weights (from repo subdir weights/)
            - "images": Sample images (from repo subdir sample_images/)
            - "notebooks": Example Jupyter notebooks (from repo subdir notebooks/)
            - "configs": YAML configs (from repo subdir configs/)
            - "all": Download everything
        asset: Alias for ``what``. Example: ``download(asset="weights")``.
        output_dir: Directory to save downloaded files. If None, uses the
            default cache directory (~/.cache/unreflectanything/).
        variant: Weights variant to download (e.g., "default").
        force: If True, re-download even if files already exist.

    Returns:
        Path to the directory where files were saved.

    Raises:
        ImportError: If huggingface_hub is not installed.
        ValueError: If neither ``what`` nor ``asset`` is set, or value is invalid.

    Example:
        >>> # Download weights (positional or keyword)
        >>> path = download("weights")
        >>> path = download(asset="weights")
        >>> # Download images or notebooks
        >>> download(asset="images")
        >>> download(asset="notebooks")
        >>> download(asset="configs")
        >>> # Download everything to custom directory
        >>> download("all", output_dir="./assets/", force=True)
    """
    resolved_what = asset if asset is not None else what
    if resolved_what is None:
        resolved_what = "weights"  # CLI default
    if resolved_what not in ("weights", "images", "notebooks", "configs", "all"):
        raise ValueError(
            f"Invalid 'what'/'asset' value: {resolved_what!r}. "
            "Must be 'weights', 'images', 'notebooks', 'configs', or 'all'."
        )
    from unreflectanything.weights import (
        download_configs,
        download_images,
        download_notebooks,
        download_weights,
        get_cache_dir,
    )

    if output_dir is None:
        output_path = get_cache_dir("weights").parent
    else:
        output_path = Path(output_dir).expanduser().resolve()

    output_path.mkdir(parents=True, exist_ok=True)

    if resolved_what == "weights":
        weights_dir = output_path / "weights"
        download_weights(output_dir=weights_dir, variant=variant, force=force)
        return weights_dir
    elif resolved_what == "images":
        images_dir = output_path / "images"
        download_images(output_dir=images_dir, force=force)
        return images_dir
    elif resolved_what == "notebooks":
        notebooks_dir = output_path / "notebooks"
        download_notebooks(output_dir=notebooks_dir, force=force)
        return notebooks_dir
    elif resolved_what == "configs":
        configs_dir = output_path / "configs"
        download_configs(output_dir=configs_dir, force=force)
        return configs_dir
    else:  # "all"
        weights_dir = output_path / "weights"
        images_dir = output_path / "images"
        notebooks_dir = output_path / "notebooks"
        configs_dir = output_path / "configs"
        download_weights(output_dir=weights_dir, variant=variant, force=force)
        download_images(output_dir=images_dir, force=force)
        download_notebooks(output_dir=notebooks_dir, force=force)
        download_configs(output_dir=configs_dir, force=force)
        return output_path


# =============================================================================
# EVALUATE
# =============================================================================

def evaluate(
    output: Union[str, PathLike, Path, "Tensor"],
    reference: Union[str, PathLike, Path, "Tensor"],
    metrics: Optional[List[str]] = None,
    mask: Optional[Union[str, PathLike, Path, "Tensor"]] = None,
) -> Dict[str, float]:
    """Compute evaluation metrics between output and reference images.

    This function computes image quality metrics comparing model outputs
    to reference (ground truth) images. It supports both file-based and
    tensor-based inputs.

    Args:
        output: Model output to evaluate. Can be:
            - Path to a single image file
            - Path to a directory of images
            - Tensor of shape [B, C, H, W]
        reference: Reference (ground truth) images. Same format as output.
        metrics: List of metrics to compute. If None, computes all available:
            - "psnr": Peak Signal-to-Noise Ratio (higher is better)
            - "ssim": Structural Similarity Index (higher is better)
            - "mse": Mean Squared Error (lower is better)
            - "deltaE2000": Color difference in LAB space (lower is better)
            - "gmsd": Gradient Magnitude Similarity Deviation (lower is better)
            - "dists": Deep Image Structure and Texture Similarity (lower is better)
        mask: Optional mask for masked evaluation. Same spatial size as images.

    Returns:
        Dictionary mapping metric names to their values.

    Raises:
        FileNotFoundError: If output or reference paths don't exist.
        ValueError: If output and reference have mismatched shapes/counts.

    Example:
        >>> # Evaluate directory of images
        >>> results = evaluate("outputs/", "references/")
        >>> print(f"PSNR: {results['psnr']:.2f} dB")
        
        >>> # Evaluate specific metrics on tensors
        >>> results = evaluate(pred_tensor, gt_tensor, metrics=["psnr", "ssim"])
    """
    # Import evaluation module
    from evaluate import evaluate_images
    
    return evaluate_images(
        output=output,
        reference=reference,
        metrics=metrics,
        mask=mask,
    )


# =============================================================================
# VERIFY (dataset or weights)
# =============================================================================

def verify(
    what: Literal["dataset", "weights"],
    path: Optional[Union[str, PathLike, Path]] = None,
    weights_path: Optional[Union[str, PathLike, Path]] = None,
    dataset_type: Optional[str] = None,
    config: Optional[Union[str, PathLike, Path]] = None,
    model_config_path: Optional[Union[str, PathLike, Path]] = None,
) -> bool:
    """Verify either dataset structure or weights integrity.

    - **dataset**: Checks that the directory at `path` has the expected
      structure for the given dataset type (or auto-detects). Requires `path`.
    - **weights**: Checks that the weights file exists and loads into the
      model with no state_dict key alignment errors (missing/unexpected keys).
      Uses `weights_path` if provided, otherwise the default cache location.

    Args:
        what: Either "dataset" or "weights".
        path: Dataset root directory (required when what="dataset").
        weights_path: Path to weights file (optional when what="weights";
            defaults to cache).
        dataset_type: Dataset type for dataset verification (e.g. "SCRREAM",
            "HOUSECAT6D", "POLARGB", "RGBP"). Auto-detect if None.
        config: Optional config for dataset verification.
        model_config_path: Optional model config YAML for weights verification
            (used if checkpoint has no embedded config).

    Returns:
        True if verification passed, False otherwise.

    Raises:
        ValueError: If what="dataset" and path is None, or if what is invalid.
        FileNotFoundError: If path (dataset) or weights file does not exist.

    Example:
        >>> verify("dataset", path="/data/SCRREAM", dataset_type="SCRREAM")
        >>> verify("weights")
        >>> verify("weights", weights_path="/path/to/weights.pt")
    """
    if what == "dataset":
        if path is None:
            raise ValueError("path is required when what='dataset'")
        return _verify_dataset_impl(
            path=Path(path).expanduser().resolve(),
            dataset_type=dataset_type,
            config=config,
        )
    elif what == "weights":
        return _verify_weights_impl(
            weights_path=weights_path,
            model_config_path=model_config_path,
        )
    else:
        raise ValueError(f"what must be 'dataset' or 'weights', got {what!r}")


def _verify_weights_impl(
    weights_path: Optional[Union[str, PathLike, Path]] = None,
    model_config_path: Optional[Union[str, PathLike, Path]] = None,
) -> bool:
    """Verify weights file exists and loads into model with no key alignment errors."""
        
    import torch
    from inference import InferenceOptions, load_model
    from unreflectanything.weights import (
        DEFAULT_WEIGHTS_FILENAME,
        get_cache_dir,
    )
    if weights_path is None:
        resolved = get_cache_dir("weights") / DEFAULT_WEIGHTS_FILENAME
    else:
        resolved = Path(weights_path).expanduser().resolve()

    if not resolved.exists():
        print(f"Weights file not found: {resolved}")
        return False
    else:
        print(f"Found weights file: {resolved}\nLoading weights and verifying key alignemnts...")

    options = InferenceOptions(
        weights_path=resolved,
        input_dir=Path("."),
        output_dir=Path("."),
        model_config_path=Path(model_config_path) if model_config_path else None,
    )

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        load_model(options, device, strict=True, verbose=False)
        print("✔️  Weights verified: loaded into model with no key alignment errors.")
        return True
    except (KeyError, RuntimeError, FileNotFoundError) as e:
        print(f"❌  Weights verification failed: {e}")
        print("Download the model weights with 'unreflect download --weights'")
        return False


def _verify_dataset_impl(
    dataset_path: Path,
    dataset_type: Optional[str] = None,
    config: Optional[Union[str, PathLike, Path]] = None,
) -> bool:
    """Internal implementation of dataset verification."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    # Try to instantiate the dataset to verify structure
    from dataset import (
        HOUSECAT6D_Dataset,
        POLARGB_Dataset,
        RGBP_Dataset,
        SCRREAM_Dataset,
    )

    dataset_classes = {
        "SCRREAM": SCRREAM_Dataset,
        "HOUSECAT6D": HOUSECAT6D_Dataset,
        "POLARGB": POLARGB_Dataset,
        "RGBP": RGBP_Dataset,
    }

    # Auto-detect dataset type if not specified
    if dataset_type is None:
        # Try each dataset class
        for name, cls in dataset_classes.items():
            try:
                ds = cls(
                    root_dir=str(dataset_path),
                    target_size=(224, 224),
                    few_images=True,
                )
                if len(ds) > 0:
                    print(f"Detected dataset type: {name}")
                    print(f"Found {len(ds)} samples")
                    return True
            except Exception:
                continue
        print("Could not auto-detect dataset type")
        return False

    # Verify specific dataset type
    dataset_type_upper = dataset_type.upper()
    if dataset_type_upper not in dataset_classes:
        print(f"Unknown dataset type: {dataset_type}")
        print(f"Available types: {list(dataset_classes.keys())}")
        return False

    cls = dataset_classes[dataset_type_upper]
    try:
        ds = cls(
            root_dir=str(dataset_path),
            target_size=(224, 224),
            few_images=True,
        )
        sample_count = len(ds)
        if sample_count > 0:
            print(f"Dataset '{dataset_type}' verified successfully!")
            print(f"Found {sample_count} samples")
            # Try to load one sample to verify data integrity
            try:
                _ = ds[0]
                print("Sample loading: OK")
            except Exception as e:
                print(f"Warning: Sample loading failed: {e}")
                return False
            return True
        else:
            print(f"Dataset '{dataset_type}' has no samples")
            return False
    except Exception as e:
        print(f"Dataset verification failed: {e}")
        return False


def verify_dataset(
    path: Union[str, PathLike, Path],
    dataset_type: Optional[str] = None,
    config: Optional[Union[str, PathLike, Path]] = None,
) -> bool:
    """Verify that a dataset has the correct structure for training/testing.

    This is a convenience wrapper around ``verify(what="dataset", path=path, ...)``.
    Kept for backward compatibility.

    Args:
        path: Path to the dataset root directory.
        dataset_type: Type of dataset to verify (e.g., "SCRREAM", "HOUSECAT6D",
            "POLARGB", "RGBP"). If None, attempts auto-detection.
        config: Optional config file with dataset specifications.

    Returns:
        True if the dataset structure is valid, False otherwise.

    Example:
        >>> is_valid = verify_dataset("/data/SCRREAM", dataset_type="SCRREAM")
    """
    return verify(
        what="dataset",
        path=path,
        dataset_type=dataset_type,
        config=config,
    )


# =============================================================================
# CITATION
# =============================================================================

def cite(format: Literal["bibtex", "apa", "mla", "ieee", "plain"] = "bibtex") -> str:
    """Get the citation for UnReflectAnything in the specified format.

    Args:
        format: Citation format. One of:
            - "bibtex": BibTeX format (default)
            - "apa": APA 7th edition format
            - "mla": MLA 9th edition format
            - "ieee": IEEE format
            - "plain": Plain text format

    Returns:
        Citation string in the requested format.

    Example:
        >>> print(cite("bibtex"))
        @article{unreflectanything2024,
            ...
        }
    """
    import importlib.resources
    from pathlib import Path

    # Try to load citations from file
    try:
        try:
            pkg = importlib.resources.files("unreflectanything")
            citations_path = pkg / "data" / "citations.txt"
            if hasattr(citations_path, 'read_text'):
                citations_text = citations_path.read_text(encoding="utf-8")
            else:
                # Fallback for older Python
                citations_path = Path(__file__).parent / "data" / "citations.txt"
                citations_text = citations_path.read_text(encoding="utf-8")
        except Exception:
            citations_path = Path(__file__).parent / "data" / "citations.txt"
            if citations_path.exists():
                citations_text = citations_path.read_text(encoding="utf-8")
            else:
                # Try assets folder in project root
                citations_path = Path(__file__).parent.parent / "assets" / "citations.txt"
                if citations_path.exists():
                    citations_text = citations_path.read_text(encoding="utf-8")
                else:
                    return _get_fallback_citation(format)
    except Exception:
        return _get_fallback_citation(format)

    # Parse citations file (sections separated by format headers)
    citations = {}
    current_format = None
    current_lines = []

    for line in citations_text.split("\n"):
        if line.startswith("[") and line.endswith("]"):
            if current_format and current_lines:
                citations[current_format] = "\n".join(current_lines).strip()
            current_format = line[1:-1].lower()
            current_lines = []
        else:
            current_lines.append(line)

    if current_format and current_lines:
        citations[current_format] = "\n".join(current_lines).strip()

    return citations.get(format.lower(), _get_fallback_citation(format))


def _get_fallback_citation(format: str) -> str:
    """Return fallback citation when citations file is not available."""
    citations = {
        "bibtex": """@article{unreflectanything2024,
    title={UnReflectAnything: Removing Specular Reflections from RGB Images},
    author={Rota, Alberto and Kiray, Mert and Karaoglu, Mert Asim and Ruhkamp, Patrick and De Momi, Elena and Navab, Nassir and Busam, Benjamin},
    journal={arXiv preprint},
    year={2024}
}""",
        "apa": """Rota, A., Kiray, M., Karaoglu, M. A., Ruhkamp, P., De Momi, E., Navab, N., & Busam, B. (2024). UnReflectAnything: Removing Specular Reflections from RGB Images. arXiv preprint.""",
        "mla": """Rota, Alberto, et al. "UnReflectAnything: Removing Specular Reflections from RGB Images." arXiv preprint, 2024.""",
        "ieee": """A. Rota, M. Kiray, M. A. Karaoglu, P. Ruhkamp, E. De Momi, N. Navab, and B. Busam, "UnReflectAnything: Removing Specular Reflections from RGB Images," arXiv preprint, 2024.""",
        "plain": """Alberto Rota, Mert Kiray, Mert Asim Karaoglu, Patrick Ruhkamp, Elena De Momi, Nassir Navab, and Benjamin Busam. UnReflectAnything: Removing Specular Reflections from RGB Images. arXiv preprint, 2024.""",
    }
    return citations.get(format.lower(), citations["bibtex"])


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "inference",
    "model",
    "UnReflectModel",
    "train",
    "test",
    "download",
    "evaluate",
    "verify_dataset",
    "cite",
]
