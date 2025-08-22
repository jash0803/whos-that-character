"""CLIP-based image attribute extractor."""

import logging
from typing import Union, List
import torch
import numpy as np
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

from src.models.base import BaseExtractor
from src.core.schema import CharacterAttributes
from config.settings import CLIP_MODEL, ATTRIBUTES, CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


class CLIPExtractor(BaseExtractor):
    """CLIP-based zero-shot attribute extraction."""
    
    def __init__(self):
        super().__init__("CLIP")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading CLIP model on {self.device}")
        
        try:
            self.model = CLIPModel.from_pretrained(CLIP_MODEL).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def _load_image(self, image_input: Union[Image.Image, str]) -> Image.Image:
        """Load and prepare image."""
        if isinstance(image_input, str):
            if image_input.startswith(('http://', 'https://')):
                response = requests.get(image_input, stream=True)
                response.raise_for_status()
                image = Image.open(response.raw)
            else:
                image = Image.open(image_input)
        else:
            image = image_input
        
        # Convert to RGB if needed 
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def _classify_attribute(self, image: Image.Image, attribute: str) -> str:
        """Classify a single attribute using CLIP."""
        options = ATTRIBUTES[attribute]
        texts = [f"a {option} person" for option in options]
        
        try:
            inputs = self.processor(
                text=texts, 
                images=image, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
            best_idx = np.argmax(probs)
            confidence = probs[best_idx]
            
            if confidence > CONFIDENCE_THRESHOLD:
                return options[best_idx]
                
        except Exception as e:
            logger.warning(f"Failed to classify {attribute}: {e}")
        
        return None
    
    def extract(self, image_input: Union[Image.Image, str]) -> CharacterAttributes:
        """Extract attributes from an image."""
        try:
            image = self._load_image(image_input)
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return CharacterAttributes()
        
        attributes = {}
        
        for attr_name in ATTRIBUTES.keys():
            try:
                result = self._classify_attribute(image, attr_name)
                attributes[attr_name] = result
            except Exception as e:
                logger.warning(f"Failed to extract {attr_name}: {e}")
                attributes[attr_name] = None
        
        return CharacterAttributes(**attributes)
    
    def batch_extract(self, images: List[Union[Image.Image, str]]) -> List[CharacterAttributes]:
        """Extract attributes from multiple images."""
        results = []
        
        for image in images:
            try:
                attrs = self.extract(image)
                results.append(attrs)
            except Exception as e:
                logger.error(f"Failed to process image: {e}")
                results.append(CharacterAttributes())
        
        return results