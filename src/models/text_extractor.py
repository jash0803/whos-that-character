"""Text-based attribute extractor."""

import logging
import torch
from transformers import pipeline

from src.models.base import BaseExtractor
from src.core.schema import CharacterAttributes
from config.settings import TEXT_MODEL, CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


class TextExtractor(BaseExtractor):
    """Text-based attribute extraction using zero-shot classification."""
    
    def __init__(self):
        super().__init__("Text")
        
        try:
            device = 0 if torch.cuda.is_available() else -1
            self.classifier = pipeline(
                "zero-shot-classification",
                model=TEXT_MODEL,
                device=device
            )
            
            # Simplified attribute mapping for text
            self.text_attributes = {
                "Age": ["child", "teen", "young adult", "middle-aged", "elderly"],
                "Gender": ["male", "female", "non-binary"],
                "Hair_Color": ["black", "brown", "blonde", "red", "white", "gray", "colorful"],
                "Eye_Color": ["brown", "blue", "green", "gray", "hazel"]
            }
            
            logger.info("Text classifier loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load text classifier: {e}")
            raise
    
    def extract(self, text: str) -> CharacterAttributes:
        """Extract attributes from text description."""
        if not text or not text.strip():
            return CharacterAttributes()
        
        attributes = {}
        
        for attr_name, labels in self.text_attributes.items():
            try:
                result = self.classifier(text, labels)
                
                if result['scores'][0] > CONFIDENCE_THRESHOLD:
                    attributes[attr_name] = result['labels'][0]
                else:
                    attributes[attr_name] = None
                    
            except Exception as e:
                logger.warning(f"Failed to classify {attr_name} from text: {e}")
                attributes[attr_name] = None
        
        # Set other attributes to None for text input
        for attr in ["Ethnicity", "Hair_Style", "Hair_Length", "Body_Type", "Dress"]:
            attributes[attr] = None
        
        return CharacterAttributes(**attributes)