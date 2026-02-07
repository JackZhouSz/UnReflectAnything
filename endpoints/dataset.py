"""Dataset utilities for UnReflectAnything endpoints."""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from torch.utils.data import Dataset


def _collect_image_paths(root: Path, extensions: Sequence[str]) -> List[Path]:
    """Collect image paths under root matching extensions (case-insensitive)."""
    lower_exts = tuple(ext.lower() for ext in extensions)
    paths = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in lower_exts
    ]
    return sorted(paths)


DEFAULT_IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)


class ImageDirDataset(Dataset):
    """
    Dataset that reads images from a directory and returns tensors.

    Each item is a tensor of shape ``(3, H, W)`` in [0, 1], optionally resized.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
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
        from PIL import Image
        from torchvision.transforms import functional as TF

        path = self.paths[idx]
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            x = TF.to_tensor(rgb)  # (3, H, W), float32, [0, 1]
        if self.target_size is not None:
            x = TF.resize(x, self.target_size, antialias=True)  # (3, H_t, W_t)
        if self.return_path:
            return x, str(path)
        return x
