import json
import logging
import sys
import time
from pathlib import Path
from typing import Set, Optional, Union
import fire

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.pipeline import CharacterPipeline
from src.core.schema import ProcessingResult, CharacterAttributes
from src.utils.danbooru_loader import DanbooruDataLoader
from config.settings import OUTPUT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DanbooruProcessor:
    """Danbooru tar file processor with resume capability."""
    
    def __init__(self, use_cache: bool = True):
        """Initialize the processor.
        
        Args:
            use_cache: Whether to use caching for the pipeline
        """
        self.use_cache = use_cache
        self.pipeline = None
    
    def _init_pipeline(self):
        """Initialize the pipeline if not already done."""
        if self.pipeline is None:
            logger.info("Initializing pipeline...")
            self.pipeline = CharacterPipeline(use_cache=self.use_cache)
            logger.info("Pipeline ready!")
    
    def stats(self, tar_path: str, output: Optional[str] = None) -> dict:
        """Show dataset statistics.
        
        Args:
            tar_path: Path to Danbooru tar file
            output: Optional output file path to check progress
            
        Returns:
            Dictionary with statistics
        """
        if not Path(tar_path).exists():
            raise FileNotFoundError(f"Tar file not found: {tar_path}")
        
        loader = DanbooruDataLoader(tar_path)
        stats = loader.get_stats()
        
        print(f"\n📊 Danbooru Tar Statistics:")
        print(f"File: {stats['tar_file']}")
        print(f"Size: {stats['file_size_mb']:.1f} MB")
        print(f"Total files: {stats['total_files']}")
        print(f"Image files: {stats['image_files']}")
        print(f"Text files: {stats['text_files']}")
        print(f"Unique IDs: {stats['unique_ids']}")
        
        # If output file exists, show progress stats
        if output and Path(output).exists():
            processed_ids = self._load_processed_ids(Path(output))
            print(f"Already processed: {len(processed_ids)}")
            print(f"Remaining: {stats['unique_ids'] - len(processed_ids)}")
            stats['processed'] = len(processed_ids)
            stats['remaining'] = stats['unique_ids'] - len(processed_ids)
        
        return stats
    
    def process(self, 
                tar_path: str,
                samples: int = 1000,
                output: Optional[str] = None,
                batch_size: int = 16,
                resume: bool = True,
                no_cache: bool = False) -> dict:
        """Process a Danbooru tar file.
        
        Args:
            tar_path: Path to Danbooru tar file
            samples: Maximum number of NEW samples to process
            output: Output file path (auto-generated if not provided)
            batch_size: Batch size for processing
            resume: Whether to resume from existing output (default: True)
            no_cache: Disable caching
            
        Returns:
            Processing results summary
        """
        if not Path(tar_path).exists():
            raise FileNotFoundError(f"Tar file not found: {tar_path}")
        
        # Override cache setting if specified
        if no_cache:
            self.use_cache = False
        
        self._init_pipeline()
        
        # Set output path
        if output is None:
            tar_name = Path(tar_path).stem
            output_path = OUTPUT_DIR / f"danbooru_{tar_name}_{int(time.time())}.jsonl"
        else:
            output_path = Path(output)
        
        logger.info(f"Loading Danbooru tar: {tar_path}")
        loader = DanbooruDataLoader(tar_path)
        
        # Show dataset stats
        stats = loader.get_stats()
        logger.info(f"Dataset stats: {stats}")
        
        # Load already processed IDs if resuming
        processed_ids = set()
        last_processed_id = None
        start_from_next = False
        
        if resume and output_path.exists():
            processed_ids = self._load_processed_ids(output_path)
            last_processed_id = self._get_last_processed_id(output_path)
            
            if processed_ids:
                logger.info(f"Resuming from existing output file")
                logger.info(f"Last processed ID: {last_processed_id}")
                logger.info(f"Total already processed: {len(processed_ids)}")
        
        results = []
        processed = 0
        skipped = 0
        errors = 0
        
        # Open file in append mode if resuming, write mode if starting fresh
        file_mode = 'a' if (resume and output_path.exists() and processed_ids) else 'w'
        
        start_time = time.time()
        
        try:
            with open(output_path, file_mode, encoding='utf-8') as output_file:
                for file_id, image, text_content in loader.load_pairs():
                    # If resuming and we haven't reached the last processed ID yet
                    if resume and processed_ids and not start_from_next:
                        if file_id in processed_ids:
                            skipped += 1
                            if file_id == last_processed_id:
                                start_from_next = True
                                logger.info(f"Reached last processed ID: {file_id}, starting fresh processing...")
                            continue
                        elif last_processed_id and file_id != last_processed_id:
                            # Skip until we find the last processed ID
                            skipped += 1
                            continue
                    
                    # Check if we've already processed this ID (safety check)
                    if file_id in processed_ids:
                        skipped += 1
                        continue
                    
                    if processed >= samples:
                        break
                    
                    try:
                        # Determine what to process
                        result = None
                        
                        if image is not None:
                            # Process image (preferred)
                            result = self.pipeline.process_image(image, item_id=file_id)
                            result.source = "image"
                            
                            # If we also have text, add it to the result
                            if text_content:
                                # Parse Danbooru tags from text
                                tags = self._parse_danbooru_tags(text_content)
                                result.tags = tags
                        
                        elif text_content is not None:
                            # Process text only
                            result = self.pipeline.process_text(text_content, item_id=file_id)
                            result.source = "text"
                            
                            # Parse tags
                            tags = self._parse_danbooru_tags(text_content)
                            result.tags = tags
                        
                        else:
                            logger.warning(f"No image or text for {file_id}")
                            continue
                        
                        if result:
                            # Convert result to dict and add tags if present
                            result_dict = result.to_dict()
                            if hasattr(result, 'tags'):
                                result_dict['tags'] = result.tags
                            
                            # Ensure the ID is properly set in the output
                            result_dict['id'] = file_id
                            
                            # Write immediately to file (streaming approach)
                            output_file.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
                            output_file.flush()  # Ensure data is written
                            
                            results.append(result)
                            processed_ids.add(file_id)  # Track processed IDs
                            
                            if result.error:
                                errors += 1
                                logger.warning(f"Error processing {file_id}: {result.error}")
                            else:
                                # Show progress for successful extractions
                                attrs = result.attributes.get_filled_attributes()
                                if attrs:
                                    logger.debug(f"✅ {file_id}: {len(attrs)} attributes")
                                else:
                                    logger.debug(f"⚠️ {file_id}: No attributes detected")
                        
                        processed += 1
                        
                        # Show progress every 50 samples
                        if processed % 50 == 0:
                            logger.info(f"Processed {processed}/{samples} new samples "
                                      f"(Skipped: {skipped}, Success: {processed-errors}, Errors: {errors})")
                        
                        # Small delay to prevent overwhelming
                        if processed % 50 == 0:
                            time.sleep(0.1)
                    
                    except Exception as e:
                        logger.error(f"Failed to process {file_id}: {e}")
                        errors += 1
                        continue
            
            total_time = time.time() - start_time
            
            # Summary
            total_in_file = len(processed_ids)  # Total unique IDs now in file
            successful = sum(1 for r in results if not r.error and not r.attributes.is_empty())
            
            print(f"\n🎉 Danbooru Processing Complete!")
            print(f"📊 Summary:")
            print(f"  Tar file: {Path(tar_path).name}")
            print(f"  New samples processed: {len(results)}")
            print(f"  Samples skipped (already done): {skipped}")
            print(f"  Total samples in output file: {total_in_file}")
            print(f"  Successful new extractions: {successful}")
            print(f"  Failed new extractions: {len(results) - successful}")
            print(f"  Output: {output_path}")
            
            # Performance stats
            pipeline_stats = self.pipeline.get_stats()
            print(f"\n⏱️ Performance:")
            print(f"  Total time: {total_time:.1f}s")
            if len(results) > 0:
                print(f"  Samples/second: {len(results)/total_time:.2f}")
            print(f"  Pipeline processed: {pipeline_stats['processed_count']}")
            
            cache_stats = pipeline_stats['cache_stats']
            if cache_stats['enabled']:
                print(f"  Cache entries: {cache_stats.get('entries', 0)}")
            
            return {
                'new_processed': len(results),
                'skipped': skipped,
                'total_in_file': total_in_file,
                'successful': successful,
                'failed': len(results) - successful,
                'errors': errors,
                'processing_time': total_time,
                'samples_per_second': len(results)/total_time if len(results) > 0 else 0,
                'output_path': str(output_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to process tar file: {e}")
            raise
    
    def resume(self, tar_path: str, samples: int = 1000, output: Optional[str] = None, **kwargs):
        """Resume processing from existing output file.
        
        This is a convenience method that calls process() with resume=True.
        
        Args:
            tar_path: Path to Danbooru tar file  
            samples: Maximum number of NEW samples to process
            output: Output file path
            **kwargs: Additional arguments passed to process()
        """
        return self.process(tar_path=tar_path, samples=samples, output=output, resume=True, **kwargs)
    
    def fresh(self, tar_path: str, samples: int = 1000, output: Optional[str] = None, **kwargs):
        """Start fresh processing (ignore existing output).
        
        This is a convenience method that calls process() with resume=False.
        
        Args:
            tar_path: Path to Danbooru tar file
            samples: Maximum number of samples to process  
            output: Output file path
            **kwargs: Additional arguments passed to process()
        """
        return self.process(tar_path=tar_path, samples=samples, output=output, resume=False, **kwargs)
    
    def _load_processed_ids(self, output_path: Path) -> Set[str]:
        """Load already processed IDs from existing output file."""
        processed_ids = set()
        
        if not output_path.exists():
            logger.info("No existing output file found - starting fresh")
            return processed_ids
        
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        if 'id' in data:
                            processed_ids.add(data['id'])
                        elif 'item_id' in data:  # fallback for different ID field names
                            processed_ids.add(data['item_id'])
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON on line {line_num}: {e}")
                        continue
            
            logger.info(f"Found {len(processed_ids)} already processed IDs in {output_path}")
            return processed_ids
            
        except Exception as e:
            logger.error(f"Error loading existing results: {e}")
            return processed_ids
    
    def _get_last_processed_id(self, output_path: Path) -> Optional[str]:
        """Get the last processed ID from the output file."""
        if not output_path.exists():
            return None
        
        try:
            with open(output_path, 'rb') as f:
                # Read from the end to find the last line efficiently
                f.seek(-2, 2)  # Go to second last byte
                while f.read(1) != b'\n':
                    f.seek(-2, 1)
                
                last_line = f.readline().decode('utf-8')
                data = json.loads(last_line.strip())
                return data.get('id') or data.get('item_id')
        
        except Exception as e:
            logger.warning(f"Could not determine last processed ID: {e}")
            return None
    
    def _parse_danbooru_tags(self, text_content: str) -> dict:
        """Parse Danbooru tags from text content."""
        try:
            # Danbooru tags are typically comma-separated
            tags = [tag.strip() for tag in text_content.split(',')]
            
            # Categorize tags (basic categorization)
            categorized = {
                "character_tags": [],
                "appearance_tags": [],
                "clothing_tags": [],
                "other_tags": []
            }
            
            # Simple keyword-based categorization
            appearance_keywords = ['hair', 'eyes', 'eye', 'skin', 'face', 'body']
            clothing_keywords = ['dress', 'shirt', 'pants', 'skirt', 'uniform', 'clothes']
            
            for tag in tags:
                tag_lower = tag.lower()
                if any(keyword in tag_lower for keyword in appearance_keywords):
                    categorized["appearance_tags"].append(tag)
                elif any(keyword in tag_lower for keyword in clothing_keywords):
                    categorized["clothing_tags"].append(tag)
                elif tag.endswith('_(character)') or 'girl' in tag_lower or 'boy' in tag_lower:
                    categorized["character_tags"].append(tag)
                else:
                    categorized["other_tags"].append(tag)
            
            return {
                "raw_tags": tags,
                "categorized": categorized,
                "tag_count": len(tags)
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse tags: {e}")
            return {"raw_tags": [], "tag_count": 0}


def main():
    """Main entry point for Fire CLI."""
    try:
        fire.Fire(DanbooruProcessor)
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        logger.info("Progress has been saved - you can resume with the same command")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()