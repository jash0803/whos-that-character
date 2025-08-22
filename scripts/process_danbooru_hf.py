import json
import logging
import sys
import time
from pathlib import Path
from typing import Set, Optional, Union, Dict, List, Any
import fire
from datasets import load_dataset, Dataset
from PIL import Image
import io

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.pipeline import CharacterPipeline
from src.core.schema import ProcessingResult, CharacterAttributes
from config.settings import OUTPUT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DanbooruProcessor:
    """Danbooru dataset processor with batch inference and resume capability."""
    
    def __init__(self, use_cache: bool = True):
        """Initialize the processor.
        
        Args:
            use_cache: Whether to use caching for the pipeline
        """
        self.use_cache = use_cache
        self.pipeline = None
        self.dataset = None
    
    def _init_pipeline(self):
        """Initialize the pipeline if not already done."""
        if self.pipeline is None:
            logger.info("Initializing pipeline...")
            self.pipeline = CharacterPipeline(use_cache=self.use_cache)
            logger.info("Pipeline ready!")
    
    def _load_dataset(self, dataset_name: str = "cagliostrolab/860k-ordered-tags", 
                      split: str = "train", streaming: bool = True):
        """Load the Hugging Face dataset."""
        if self.dataset is None:
            logger.info(f"Loading dataset: {dataset_name}")
            self.dataset = load_dataset(dataset_name, split=split, streaming=streaming)
            logger.info(f"Dataset loaded with {len(self.dataset) if not streaming else 'streaming'} samples")
        return self.dataset
    
    def stats(self, 
              dataset_name: str = "cagliostrolab/860k-ordered-tags",
              split: str = "train",
              output: Optional[str] = None) -> dict:
        """Show dataset statistics.
        
        Args:
            dataset_name: Hugging Face dataset name
            split: Dataset split to use
            output: Optional output file path to check progress
            
        Returns:
            Dictionary with statistics
        """
        dataset = self._load_dataset(dataset_name, split)
        
        # Get basic dataset info
        total_samples = len(dataset)
        sample = dataset[0]
        
        print(f"\n📊 Danbooru Dataset Statistics:")
        print(f"Dataset: {dataset_name}")
        print(f"Split: {split}")
        print(f"Total samples: {total_samples:,}")
        print(f"Sample keys: {list(sample.keys())}")
        
        # Show sample structure
        if 'image' in sample:
            img = sample['image']
            if hasattr(img, 'size'):
                print(f"Sample image size: {img.size}")
        
        if 'general' in sample:
            print(f"Sample tags count: {len(sample['general']) if sample['general'] else 0}")
        
        stats = {
            'dataset_name': dataset_name,
            'split': split,
            'total_samples': total_samples,
            'sample_keys': list(sample.keys())
        }
        
        # If output file exists, show progress stats
        if output and Path(output).exists():
            processed_ids = self._load_processed_ids(Path(output))
            print(f"Already processed: {len(processed_ids):,}")
            print(f"Remaining: {total_samples - len(processed_ids):,}")
            stats['processed'] = len(processed_ids)
            stats['remaining'] = total_samples - len(processed_ids)
        
        return stats
    
    def _batch_process_images(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Process a batch of images using the pipeline.
        
        Args:
            batch: Dictionary with lists of batch data
            
        Returns:
            Dictionary with processed results
        """
        batch_size = len(batch['image'])
        results = []
        
        # Process each item in the batch
        for i in range(batch_size):
            try:
                # Get the image
                image = batch['image'][i]
                item_id = batch.get('id', [f"item_{i}"])[i]
                
                # Convert PIL image if needed
                if isinstance(image, dict) and 'bytes' in image:
                    image = Image.open(io.BytesIO(image['bytes']))
                
                # Process with pipeline
                result = self.pipeline.process_image(image, item_id=str(item_id))
                result.source = "image"
                
                # Add tags if available
                if 'general' in batch and batch['general'][i]:
                    tags = self._parse_danbooru_tags(batch['general'][i])
                    result.tags = tags
                elif 'tag_string_general' in batch and batch['tag_string_general'][i]:
                    tags = self._parse_danbooru_tags(batch['tag_string_general'][i])
                    result.tags = tags
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing item {i} in batch: {e}")
                # Create error result
                error_result = ProcessingResult(
                    item_id=str(batch.get('id', [f"item_{i}"])[i]),
                    attributes=CharacterAttributes(),
                    source="image",
                    error=str(e)
                )
                results.append(error_result)
        
        # Convert results to batch format
        batch_results = {
            'processing_results': results,
            'processed_ids': [r.item_id for r in results],
            'success_count': sum(1 for r in results if not r.error),
            'error_count': sum(1 for r in results if r.error)
        }
        
        return batch_results
    
    def process(self, 
                samples: int = 1000,
                dataset_name: str = "cagliostrolab/860k-ordered-tags",
                split: str = "train",
                output: Optional[str] = None,
                batch_size: int = 16,
                resume: bool = True,
                no_cache: bool = False,
                start_idx: int = 0,
                streaming: bool = False) -> dict:
        """Process the Danbooru dataset with batch inference.
        
        Args:
            samples: Maximum number of NEW samples to process
            dataset_name: Hugging Face dataset name
            split: Dataset split to use
            output: Output file path (auto-generated if not provided)
            batch_size: Batch size for processing
            resume: Whether to resume from existing output (default: True)
            no_cache: Disable caching
            start_idx: Start index in dataset (useful for manual resuming)
            streaming: Use streaming dataset (for very large datasets)
            
        Returns:
            Processing results summary
        """
        # Override cache setting if specified
        if no_cache:
            self.use_cache = False
        
        self._init_pipeline()
        dataset = self._load_dataset(dataset_name, split, streaming)
        
        # Set output path
        if output is None:
            output_path = OUTPUT_DIR / f"danbooru_{dataset_name.replace('/', '_')}_{int(time.time())}.jsonl"
        else:
            output_path = Path(output)
        
        # Load already processed IDs if resuming
        processed_ids = set()
        if resume and output_path.exists():
            processed_ids = self._load_processed_ids(output_path)
            if processed_ids:
                logger.info(f"Resuming from existing output file")
                logger.info(f"Total already processed: {len(processed_ids)}")
        
        # Determine processing range
        if streaming:
            # For streaming, we'll process sequentially
            dataset_slice = dataset.skip(start_idx).take(samples)
            logger.info(f"Processing {samples} samples starting from index {start_idx} (streaming)")
        else:
            # For non-streaming, determine end index
            total_samples = len(dataset)
            end_idx = min(start_idx + samples, total_samples)
            
            # If resuming, find the actual start index
            if resume and processed_ids and not start_idx:
                # Find the last processed index (this is approximate)
                start_idx = len(processed_ids)
                end_idx = min(start_idx + samples, total_samples)
            
            dataset_slice = dataset.select(range(start_idx, end_idx))
            logger.info(f"Processing samples {start_idx:,} to {end_idx:,} ({len(dataset_slice):,} samples)")
        
        # Filter out already processed samples if resuming
        if resume and processed_ids:
            def not_processed(example):
                item_id = str(example.get('id', example.get('image_id', 'unknown')))
                return item_id not in processed_ids
            
            dataset_slice = dataset_slice.filter(not_processed)
            logger.info(f"After filtering processed samples: {len(dataset_slice)} remain")
        
        if len(dataset_slice) == 0:
            logger.info("No new samples to process!")
            return {'new_processed': 0, 'message': 'All samples already processed'}
        
        # Process in batches using datasets.map()
        logger.info(f"Starting batch processing with batch_size={batch_size}")
        start_time = time.time()
        
        # Open output file
        file_mode = 'a' if (resume and output_path.exists()) else 'w'
        processed_count = 0
        total_success = 0
        total_errors = 0
        
        try:
            with open(output_path, file_mode, encoding='utf-8') as output_file:
                # Process dataset in batches
                for batch_idx in range(0, len(dataset_slice), batch_size):
                    batch_end = min(batch_idx + batch_size, len(dataset_slice))
                    batch_data = dataset_slice.select(range(batch_idx, batch_end))
                    
                    # Convert to batch format for processing
                    batch_dict = {}
                    for key in batch_data.features.keys():
                        batch_dict[key] = [batch_data[i][key] for i in range(len(batch_data))]
                    
                    # Process batch
                    batch_results = self._batch_process_images(batch_dict)
                    
                    # Write results to file
                    for result in batch_results['processing_results']:
                        # Convert result to dict
                        result_dict = result.to_dict()
                        if hasattr(result, 'tags'):
                            result_dict['tags'] = result.tags
                        
                        # Write to file
                        output_file.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
                    
                    output_file.flush()  # Ensure data is written
                    
                    # Update counters
                    processed_count += len(batch_results['processing_results'])
                    total_success += batch_results['success_count']
                    total_errors += batch_results['error_count']
                    
                    # Log progress
                    if batch_idx % (batch_size * 5) == 0:  # Log every 5 batches
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        logger.info(f"Processed {processed_count:,} samples "
                                  f"(Success: {total_success}, Errors: {total_errors}, "
                                  f"Rate: {rate:.1f} samples/sec)")
                    
                    # Small delay to prevent overwhelming
                    time.sleep(0.01)
        
            total_time = time.time() - start_time
            
            # Summary
            print(f"\n🎉 Danbooru Processing Complete!")
            print(f"📊 Summary:")
            print(f"  Dataset: {dataset_name}")
            print(f"  New samples processed: {processed_count:,}")
            print(f"  Successful extractions: {total_success:,}")
            print(f"  Failed extractions: {total_errors:,}")
            print(f"  Output: {output_path}")
            
            # Performance stats
            if hasattr(self.pipeline, 'get_stats'):
                pipeline_stats = self.pipeline.get_stats()
                print(f"\n⏱️ Performance:")
                print(f"  Total time: {total_time:.1f}s")
                print(f"  Samples/second: {processed_count/total_time:.2f}")
                print(f"  Pipeline processed: {pipeline_stats.get('processed_count', 0)}")
                
                cache_stats = pipeline_stats.get('cache_stats', {})
                if cache_stats.get('enabled'):
                    print(f"  Cache entries: {cache_stats.get('entries', 0)}")
            
            return {
                'new_processed': processed_count,
                'successful': total_success,
                'failed': total_errors,
                'processing_time': total_time,
                'samples_per_second': processed_count/total_time if total_time > 0 else 0,
                'output_path': str(output_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to process dataset: {e}")
            raise
    
    def resume(self, samples: int = 1000, output: str = None, **kwargs):
        """Resume processing from existing output file.
        
        Args:
            samples: Maximum number of NEW samples to process
            output: Output file path
            **kwargs: Additional arguments passed to process()
        """
        return self.process(samples=samples, output=output, resume=True, **kwargs)
    
    def fresh(self, samples: int = 1000, output: str = None, **kwargs):
        """Start fresh processing (ignore existing output).
        
        Args:
            samples: Maximum number of samples to process  
            output: Output file path
            **kwargs: Additional arguments passed to process()
        """
        return self.process(samples=samples, output=output, resume=False, **kwargs)
    
    def stream_process(self, samples: int = 1000, batch_size: int = 16, **kwargs):
        """Process dataset in streaming mode for very large datasets.
        
        Args:
            samples: Maximum number of samples to process
            batch_size: Batch size for processing
            **kwargs: Additional arguments passed to process()
        """
        return self.process(samples=samples, batch_size=batch_size, streaming=True, **kwargs)
    
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
                        if 'item_id' in data:
                            processed_ids.add(data['item_id'])
                        elif 'id' in data:
                            processed_ids.add(str(data['id']))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON on line {line_num}: {e}")
                        continue
            
            logger.info(f"Found {len(processed_ids)} already processed IDs in {output_path}")
            return processed_ids
            
        except Exception as e:
            logger.error(f"Error loading existing results: {e}")
            return processed_ids
    
    def _parse_danbooru_tags(self, tags_input: Union[str, List[str]]) -> dict:
        """Parse Danbooru tags from various input formats."""
        try:
            # Handle different input formats
            if isinstance(tags_input, list):
                tags = tags_input
            elif isinstance(tags_input, str):
                # Tags might be comma-separated or space-separated
                if ',' in tags_input:
                    tags = [tag.strip() for tag in tags_input.split(',')]
                else:
                    tags = [tag.strip() for tag in tags_input.split()]
            else:
                tags = []
            
            # Filter out empty tags
            tags = [tag for tag in tags if tag.strip()]
            
            # Categorize tags
            categorized = {
                "character_tags": [],
                "appearance_tags": [],
                "clothing_tags": [],
                "other_tags": []
            }
            
            # Keywords for categorization
            appearance_keywords = ['hair', 'eyes', 'eye', 'skin', 'face', 'body', 'ears']
            clothing_keywords = ['dress', 'shirt', 'pants', 'skirt', 'uniform', 'clothes', 'jacket', 'coat']
            
            for tag in tags:
                tag_lower = tag.lower()
                if any(keyword in tag_lower for keyword in appearance_keywords):
                    categorized["appearance_tags"].append(tag)
                elif any(keyword in tag_lower for keyword in clothing_keywords):
                    categorized["clothing_tags"].append(tag)
                elif ('girl' in tag_lower or 'boy' in tag_lower or 
                      tag.endswith('_(character)') or 'character' in tag_lower):
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