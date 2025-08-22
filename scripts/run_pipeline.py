import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.pipeline import CharacterPipeline
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_single_image(image_path: str, pipeline: CharacterPipeline):
    """Process a single image file."""
    try:
        image = Image.open(image_path)
        result = pipeline.process_image(image, item_id=Path(image_path).stem)
        
        print(f"\n📷 Image: {image_path}")
        print("=" * 50)
        
        if result.error:
            print(f"❌ Error: {result.error}")
        else:
            attrs = result.attributes.get_filled_attributes()
            if attrs:
                for key, value in attrs.items():
                    print(f"{key.replace('_', ' '):<15}: {value}")
            else:
                print("⚠️ No attributes detected")
        
        print(f"⏱️ Processing time: {result.processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to process {image_path}: {e}")


def process_single_text(text: str, pipeline: CharacterPipeline):
    """Process a single text description."""
    try:
        result = pipeline.process_text(text, item_id="text_input")
        
        print(f"\n📝 Text: {text[:50]}...")
        print("=" * 50)
        
        if result.error:
            print(f"❌ Error: {result.error}")
        else:
            attrs = result.attributes.get_filled_attributes()
            if attrs:
                for key, value in attrs.items():
                    print(f"{key.replace('_', ' '):<15}: {value}")
            else:
                print("⚠️ No attributes detected")
        
        print(f"⏱️ Processing time: {result.processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to process text: {e}")


def process_directory(directory: str, pipeline: CharacterPipeline):
    """Process all images in a directory."""
    dir_path = Path(directory)
    
    if not dir_path.exists():
        logger.error(f"Directory does not exist: {directory}")
        return
    
    # Find image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        f for f in dir_path.rglob("*") 
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        logger.warning(f"No image files found in {directory}")
        return
    
    print(f"Found {len(image_files)} images to process...")
    
    results = []
    for image_path in image_files:
        try:
            image = Image.open(image_path)
            result = pipeline.process_image(image, item_id=image_path.stem)
            results.append(result)
            
            if not result.error and result.attributes.get_filled_attributes():
                print(f"✅ {image_path.name}")
            else:
                print(f"⚠️ {image_path.name}")
                
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
    
    # Save results
    output_path = pipeline.save_results(results)
    print(f"\n💾 Results saved to: {output_path}")
    
    # Show summary
    successful = sum(1 for r in results if not r.error and not r.attributes.is_empty())
    print(f"\n📊 Summary:")
    print(f"Total processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Character Attribute Extraction Pipeline")
    parser.add_argument("--image", type=str, help="Path to single image file")
    parser.add_argument("--text", type=str, help="Text description to process")
    parser.add_argument("--directory", type=str, help="Directory of images to process")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    
    args = parser.parse_args()
    
    if not any([args.image, args.text, args.directory]):
        parser.error("Must specify --image, --text, or --directory")
    
    try:
        # Initialize pipeline
        logger.info("Initializing pipeline...")
        pipeline = CharacterPipeline(use_cache=not args.no_cache)
        logger.info("Pipeline ready!")
        
        # Process based on arguments
        if args.image:
            process_single_image(args.image, pipeline)
        elif args.text:
            process_single_text(args.text, pipeline)
        elif args.directory:
            process_directory(args.directory, pipeline)
        
        # Show stats
        stats = pipeline.get_stats()
        print(f"\n📊 Pipeline Stats:")
        print(f"Total processed: {stats['processed_count']}")
        
        cache_stats = stats['cache_stats']
        if cache_stats['enabled']:
            print(f"Cache entries: {cache_stats.get('entries', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()