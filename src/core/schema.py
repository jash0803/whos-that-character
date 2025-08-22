"""Data schemas for character attributes."""

import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any


@dataclass
class CharacterAttributes:
    """Character attributes schema."""
    Age: Optional[str] = None
    Gender: Optional[str] = None
    Ethnicity: Optional[str] = None
    Hair_Style: Optional[str] = None
    Hair_Color: Optional[str] = None
    Hair_Length: Optional[str] = None
    Eye_Color: Optional[str] = None
    Body_Type: Optional[str] = None
    Dress: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CharacterAttributes':
        """Create from dictionary."""
        # Filter out keys not in the dataclass
        valid_keys = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)
    
    def is_empty(self) -> bool:
        """Check if all attributes are None."""
        return all(value is None for value in self.to_dict().values())
    
    def get_filled_attributes(self) -> Dict[str, str]:
        """Get only non-None attributes."""
        return {k: v for k, v in self.to_dict().items() if v is not None}


@dataclass
class ProcessingResult:
    """Result of processing a single item."""
    id: str
    attributes: CharacterAttributes
    source: str = "unknown"
    processing_time: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "attributes": self.attributes.to_dict(),
            "source": self.source,
            "processing_time": self.processing_time,
            "error": self.error
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)