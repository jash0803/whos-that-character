"""Main pipeline for character attribute extraction."""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Union, Optional

from PIL import Image
from datasets import load_dataset

from src.models.clip_extractor import CLIPExtractor
from src.models.text_extractor import TextExtractor
from src.core.schema import CharacterAttributes, ProcessingResult
from src.core.cache import SimpleCache
from config.settings import MAX_WORKERS, DEFAULT_BATCH_SIZE, OUTPUT_DIR

logger = logging.getLogger(__name__)


class CharacterPipeline:
    """Main pipeline for processing character data."""
    
    def __init__(self, use_cache: bool = True):
        self.image_extractor = CLIPExtractor()
        self.text_extractor = TextExtractor()
        self.cache = SimpleCache() if use_cache else None
        self.processed_count = 0
    
    def process_image(self, image: Union[Image.Image, str], item_id: str = None) -> ProcessingResult:
        """Process a single image."""
        start_time = time.time()
        item_id = item_id or f"img_{self.processed_count}"
        
        # Check cache
        cache_key = str(image) if isinstance(image, str) else f"image_{item_id}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return ProcessingResult(
                    id=item_id,
                    attributes=cached,
                    processing_time=time.time() - start_time
                )
        
        try:
            attributes = self.image_extractor.extract(image)
            
            # Cache result
            if self.cache:
                self.cache.set(cache_key, attributes)
            
            self.processed_count += 1
            
            return ProcessingResult(
                id=item_id,
                attributes=attributes,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Failed to process image {item_id}: {e}")
            return ProcessingResult(
                id=item_id,
                attributes=CharacterAttributes(),
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    def process_text(self, text: str, item_id: str = None) -> ProcessingResult:
        """Process a text description."""
        start_time = time.time()
        item_id = item_id or f"text_{self.processed_count}"
        
        # Check cache
        if self.cache:
            cached = self.cache.get(text)
            if cached:
                return ProcessingResult(
                    id=item_id,
                    attributes=cached,
                    processing_time=time.time() - start_time
                )
        
        try:
            attributes = self.text_extractor.extract(text)
            
            # Cache result
            if self.cache:
                self.cache.set(text, attributes)
            
            self.processed_count += 1
            
            return ProcessingResult(
                id=item_id,
                attributes=attributes,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Failed to process text {item_id}: {e}")
            return ProcessingResult(
                id=item_id,
                attributes=CharacterAttributes(),
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    def process_batch(self, 
                     items: List[Union[Image.Image, str]], 
                     input_type: str = "image",
                     batch_size: int = DEFAULT_BATCH_SIZE) -> List[ProcessingResult]:
        """Process multiple items in parallel."""
        
        results = []
        
        # Process in chunks
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                if input_type == "image":
                    futures = {
                        executor.submit(self.process_image, item, f"batch_{i}_{j}"): j 
                        for j, item in enumerate(batch)
                    }
                else:
                    futures = {
                        executor.submit(self.process_text, item, f"batch_{i}_{j}"): j 
                        for j, item in enumerate(batch)
                    }
                
                batch_results = []
                for future in as_completed(futures):
                    result = future.result()
                    batch_results.append(result)
                
                results.extend(batch_results)
                logger.info(f"Processed batch {i//batch_size + 1}, total: {len(results)}")
        
        return results
    
    def save_results(self, results: List[ProcessingResult], output_path: Path = None):
        """Save results to JSONL file."""
        if output_path is None:
            output_path = OUTPUT_DIR / f"results_{int(time.time())}.jsonl"
        
        try:
            with open(output_path, 'w') as f:
                for result in results:
                    f.write(json.dumps(result.to_dict()) + '\n')
            
            logger.info(f"Results saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        stats = {
            "processed_count": self.processed_count,
            "cache_stats": self.cache.stats() if self.cache else {"enabled": False}
        }
        return stats