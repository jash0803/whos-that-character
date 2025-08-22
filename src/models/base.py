"""Base classes for attribute extractors."""

from abc import ABC, abstractmethod
from typing import List, Any
from src.core.schema import CharacterAttributes


class BaseExtractor(ABC):
    """Abstract base class for all extractors."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def extract(self, input_data: Any) -> CharacterAttributes:
        """Extract attributes from a single input."""
        pass
    
    def batch_extract(self, inputs: List[Any]) -> List[CharacterAttributes]:
        """Extract attributes from multiple inputs."""
        return [self.extract(inp) for inp in inputs]
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"