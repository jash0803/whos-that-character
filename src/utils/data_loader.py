"""Data loading utilities."""

import logging
from pathlib import Path
from typing import List, Iterator, Union, Tuple
from PIL import Image
import tarfile
import io
import os
from typing import Optional

logger = logging.getLogger(__name__)

def load_images_from_directory(directory: Union[str, Path]) -> Iterator[Tuple[str, Image.Image]]:
    """Load all images from a directory."""
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    for file_path in dir_path.rglob("*"):
        if file_path.suffix.lower() in image_extensions:
            try:
                image = Image.open(file_path)
                yield str(file_path), image
            except Exception as e:
                logger.warning(f"Failed to load image {file_path}: {e}")
                continue

def batch_iterator(items: List, batch_size: int) -> Iterator[List]:
    """Split items into batches."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

class LocalTarLoader:
    """
    Streams records (image + text) from a tar shard.
    Expects files like:
      .../danbooru_<id>_<hash>.jpg
      .../danbooru_<id>_<hash>.txt
    """

    def __init__(self, tar_path: str, sample_limit: Optional[int] = None):
        self.tar_path = tar_path
        self.sample_limit = sample_limit

    def stream(self):
        samples = {}
        count = 0
        with tarfile.open(self.tar_path, "r") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = os.path.basename(member.name)
                base, ext = os.path.splitext(name)
                file_id = base  # e.g. danbooru_7156653_72311d849ede...

                # Extract file bytes
                f = tar.extractfile(member)
                if f is None:
                    continue
                data = f.read()

                # Group by ID
                if file_id not in samples:
                    samples[file_id] = {}

                if ext.lower() in [".jpg", ".png", ".jpeg"]:
                    try:
                        img = Image.open(io.BytesIO(data)).convert("RGB")
                        samples[file_id]["image"] = img
                    except Exception:
                        continue
                elif ext.lower() == ".txt":
                    try:
                        txt = data.decode("utf-8", errors="ignore").strip()
                        samples[file_id]["text"] = txt
                    except Exception:
                        continue

                # If we have both image + text, yield sample
                if "image" in samples[file_id] and "text" in samples[file_id]:
                    yield file_id, samples[file_id]
                    count += 1
                    del samples[file_id]  # free memory

                    if self.sample_limit and count >= self.sample_limit:
                        break
