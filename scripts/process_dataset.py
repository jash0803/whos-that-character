import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets import load_dataset
from src.core.pipeline import CharacterPipeline
from config.settings import DEFAULT_DATASET, OUTPUT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_dataset(dataset_name: str, 
                   num_samples: int, 
                   batch_size: int,
                   pipeline: CharacterPipeline,
                   output_path: Path = None):
    """Process a HuggingFace dataset."""
    
    if output_path is None:
        output_path = OUTPUT_DIR / f"dataset_results_{int(time.time())}.jsonl"
    
    logger.info(f"Loading dataset: {dataset_name}")
    logger.info(f"Processing {num_samples} samples in batches of {batch_size}")
    
    try:
        # Load dataset with streaming for large datasets
        dataset = load_dataset(dataset_name, split="train", streaming=True)
        
        results = []
        processed = 0
        errors = 0
        
        # Process in batches
        batch = []
        
        for sample_idx, sample in enumerate(dataset.take(num_samples)):
            # Add sample to batch
            batch.append((sample_idx, sample))
            
            # Process batch when full
            if len(batch) >= batch_size:
                batch_results = process_batch(batch, pipeline)
                results.extend(batch_results)
                
                # Count successful vs failed
                batch_success = sum(1 for r in batch_results if not r.error)
                batch_errors = len(batch_results) - batch_success
                
                processed += len(batch_results)
                errors += batch_errors
                
                # Save incrementally
                save_results_incremental(results, output_path)
                
                # Log progress
                logger.info(f"Processed {processed}/{num_samples} samples "
                          f"(Success: {processed-errors}, Errors: {errors})")
                
                # Clear batch
                batch = []
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
        
        # Process remaining samples
        if batch:
            batch_results = process_batch(batch, pipeline)
            results.extend(batch_results)
            processed += len(batch_results)
        
        # Final save
        save_results_incremental(results, output_path)
        
        # Summary
        successful = sum(1 for r in results if not r.error and not r.attributes.is_empty())
        
        print(f"\n🎉 Dataset Processing Complete!")
        print(f"📊 Summary:")
        print(f"  Total samples: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {len(results) - successful}")
        print(f"  Output: {output_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to process dataset: {e}")
        raise


def process_batch(batch, pipeline):
    """Process a batch of samples."""
    results = []
    
    for sample_idx, sample in batch:
        try:
            item_id = f"sample_{sample_idx}"
            
            # Try to process image first, then text
            if 'image' in sample and sample['image'] is not None:
                result = pipeline.process_image(sample['image'], item_id)
                result.source = "image"
            elif 'text' in sample and sample['text'] is not None:
                result = pipeline.process_text(sample['text'], item_id)
                result.source = "text"
            elif 'caption' in sample and sample['caption'] is not None:
                result = pipeline.process_text(sample['caption'], item_id)
                result.source = "caption"
            else:
                # Skip samples without usable data
                logger.warning(f"Sample {sample_idx} has no processable image or text")
                continue
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to process sample {sample_idx}: {e}")
            # Add error result
            from src.core.schema import CharacterAttributes, ProcessingResult
            error_result = ProcessingResult(
                id=f"sample_{sample_idx}",
                attributes=CharacterAttributes(),
                error=str(e)
            )
            results.append(error_result)
    
    return results


def save_results_incremental(results, output_path):
    """Save results incrementally to avoid data loss."""
    try:
        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result.to_dict()) + '\n')
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Process HuggingFace Dataset")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                       help="HuggingFace dataset name")
    parser.add_argument("--samples", type=int, default=100,
                       help="Number of samples to process")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for processing")
    parser.add_argument("--output", type=str,
                       help="Output file path")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable caching")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.samples <= 0:
        parser.error("Number of samples must be positive")
    
    if args.batch_size <= 0:
        parser.error("Batch size must be positive")
    
    try:
        # Initialize pipeline
        logger.info("Initializing pipeline...")
        pipeline = CharacterPipeline(use_cache=not args.no_cache)
        logger.info("Pipeline ready!")
        
        # Set output path
        output_path = Path(args.output) if args.output else None
        
        # Process dataset
        start_time = time.time()
        results = process_dataset(
            dataset_name=args.dataset,
            num_samples=args.samples,
            batch_size=args.batch_size,
            pipeline=pipeline,
            output_path=output_path
        )
        
        total_time = time.time() - start_time
        
        # Final statistics
        stats = pipeline.get_stats()
        print(f"\n⏱️ Performance:")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Samples/second: {len(results)/total_time:.2f}")
        print(f"  Pipeline processed: {stats['processed_count']}")
        
        cache_stats = stats['cache_stats']
        if cache_stats['enabled']:
            print(f"  Cache hits: {cache_stats.get('entries', 0)}")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()