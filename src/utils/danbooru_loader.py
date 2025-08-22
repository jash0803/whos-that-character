"""Data loader for Danbooru dataset files."""

import logging
import tarfile
from pathlib import Path
from typing import Iterator, Tuple, Optional, Dict, Any
from PIL import Image
import io

logger = logging.getLogger(__name__)


class DanbooruDataLoader:
    """Loader for Danbooru tar files with image/text pairs."""
    
    def __init__(self, tar_path: str):
        """Initialize loader with tar file path."""
        self.tar_path = Path(tar_path)
        if not self.tar_path.exists():
            raise FileNotFoundError(f"Tar file not found: {tar_path}")
        
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        self.text_extensions = {'.txt'}
    
    def load_pairs(self) -> Iterator[Tuple[str, Image.Image, Optional[str]]]:
        """
        Load image-text pairs from tar file.
        
        Yields:
            Tuple of (id, image, text_content)
        """
        try:
            with tarfile.open(self.tar_path, 'r') as tar:
                # Get all members
                members = tar.getmembers()
                
                # Group files by ID (danbooru_XXXXXXX)
                file_groups = self._group_files_by_id(members)
                
                for file_id, files in file_groups.items():
                    try:
                        # Load image and text for this ID
                        image = None
                        text_content = None
                        
                        for member in files:
                            file_ext = Path(member.name).suffix.lower()
                            
                            if file_ext in self.image_extensions and image is None:
                                # Load image
                                image_file = tar.extractfile(member)
                                if image_file:
                                    image_data = image_file.read()
                                    image = Image.open(io.BytesIO(image_data))
                            
                            elif file_ext in self.text_extensions and text_content is None:
                                # Load text
                                text_file = tar.extractfile(member)
                                if text_file:
                                    text_content = text_file.read().decode('utf-8', errors='ignore')
                        
                        # Yield if we have at least an image or text
                        if image is not None or text_content is not None:
                            yield file_id, image, text_content
                        else:
                            logger.warning(f"No valid image or text found for {file_id}")
                    
                    except Exception as e:
                        logger.error(f"Failed to process {file_id}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Failed to open tar file {self.tar_path}: {e}")
            raise
    
    def _group_files_by_id(self, members) -> Dict[str, list]:
        """Group tar members by danbooru ID."""
        groups = {}
        
        for member in members:
            if member.isfile():
                # Extract ID from filename: danbooru_7156653_hash.ext -> danbooru_7156653
                filename = Path(member.name).name
                if filename.startswith('danbooru_'):
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        file_id = f"{parts[0]}_{parts[1]}"  # danbooru_XXXXXXX
                        
                        if file_id not in groups:
                            groups[file_id] = []
                        groups[file_id].append(member)
        
        return groups
    
    def get_file_count(self) -> int:
        """Get total number of files in tar."""
        try:
            with tarfile.open(self.tar_path, 'r') as tar:
                return len([m for m in tar.getmembers() if m.isfile()])
        except Exception as e:
            logger.error(f"Failed to count files: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        try:
            with tarfile.open(self.tar_path, 'r') as tar:
                members = [m for m in tar.getmembers() if m.isfile()]
                
                image_count = sum(1 for m in members 
                                if Path(m.name).suffix.lower() in self.image_extensions)
                text_count = sum(1 for m in members 
                               if Path(m.name).suffix.lower() in self.text_extensions)
                
                file_groups = self._group_files_by_id(members)
                
                return {
                    "tar_file": str(self.tar_path),
                    "total_files": len(members),
                    "image_files": image_count,
                    "text_files": text_count,
                    "unique_ids": len(file_groups),
                    "file_size_mb": self.tar_path.stat().st_size / (1024 * 1024)
                }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}