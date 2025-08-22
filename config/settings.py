"""Configuration settings for the character pipeline."""

import os
from pathlib import Path
from typing import Dict, List

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache" 
OUTPUT_DIR = DATA_DIR / "outputs"
SAMPLE_DIR = DATA_DIR / "samples"

# Create directories
for dir_path in [DATA_DIR, CACHE_DIR, OUTPUT_DIR, SAMPLE_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Model settings
CLIP_MODEL = "openai/clip-vit-base-patch32"
TEXT_MODEL = "facebook/bart-large-mnli"

# Processing settings
DEFAULT_BATCH_SIZE = 8
MAX_WORKERS = 4
CONFIDENCE_THRESHOLD = 0.3

# Cache settings
CACHE_DB_PATH = CACHE_DIR / "character_cache.db"
ENABLE_CACHING = True

# Attribute definitions
ATTRIBUTES: Dict[str, List[str]] = {
    "Age": ["child", "teen", "young adult", "middle-aged", "elderly"],
    "Gender": ["male", "female", "non-binary"],
    "Ethnicity": ["Asian", "African", "Caucasian", "Hispanic", "Middle Eastern"],
    "Hair_Style": ["straight", "curly", "wavy", "ponytail", "bun", "braided"],
    "Hair_Color": ["black", "brown", "blonde", "red", "white", "gray", "colorful"],
    "Hair_Length": ["bald", "short", "medium", "long"],
    "Eye_Color": ["brown", "blue", "green", "gray", "hazel", "amber"],
    "Body_Type": ["slim", "average", "muscular", "curvy", "heavy"],
    "Dress": ["casual", "formal", "traditional", "uniform", "fantasy", "modern"]
}

# Dataset settings
DEFAULT_DATASET = "cagliostrolab/860k-ordered-tags"
DEFAULT_SAMPLES = 100

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"