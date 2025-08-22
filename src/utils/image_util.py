"""Image processing utilities."""

import logging
from typing import Union, Tuple
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

def resize_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    """Resize image while maintaining aspect ratio."""
    width, height = image.size
    
    if max(width, height) <= max_size:
        return image
    
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def validate_image(image: Union[Image.Image, str]) -> bool:
    """Validate if image is processable."""
    try:
        if isinstance(image, str):
            img = Image.open(image)
        else:
            img = image
        
        # Check if image has valid dimensions
        width, height = img.size
        if width < 32 or height < 32:
            return False
        
        # Check if image has valid mode
        if img.mode not in ['RGB', 'RGBA', 'L']:
            return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Image validation failed: {e}")
        return False

def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess image for model input."""
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize if too large
    image = resize_image(image, max_size=512)
    
    return image