"""Gradio web interface for the character pipeline."""

import logging
import gradio as gr
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.pipeline import CharacterPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline = None

def initialize_pipeline():
    """Initialize the pipeline once."""
    global pipeline
    if pipeline is None:
        try:
            logger.info("Initializing pipeline...")
            pipeline = CharacterPipeline(use_cache=True)
            logger.info("Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise

def process_image(image):
    """Process uploaded image."""
    if image is None:
        return "❌ Please upload an image."
    
    try:
        result = pipeline.process_image(image)
        
        if result.error:
            return f"❌ Error: {result.error}"
        
        # Format output nicely
        attrs = result.attributes.get_filled_attributes()
        if not attrs:
            return "⚠️ No attributes detected with sufficient confidence."
        
        output = "✅ **Detected Attributes:**\n\n"
        for key, value in attrs.items():
            output += f"• **{key.replace('_', ' ')}**: {value}\n"
        
        output += f"\n⏱️ Processing time: {result.processing_time:.2f}s"
        
        return output
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return f"❌ Processing failed: {str(e)}"

def process_text(text):
    """Process text description."""
    if not text or not text.strip():
        return "❌ Please enter a character description."
    
    try:
        result = pipeline.process_text(text)
        
        if result.error:
            return f"❌ Error: {result.error}"
        
        # Format output nicely
        attrs = result.attributes.get_filled_attributes()
        if not attrs:
            return "⚠️ No attributes detected with sufficient confidence."
        
        output = "✅ **Detected Attributes:**\n\n"
        for key, value in attrs.items():
            output += f"• **{key.replace('_', ' ')}**: {value}\n"
        
        output += f"\n⏱️ Processing time: {result.processing_time:.2f}s"
        
        return output
        
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return f"❌ Processing failed: {str(e)}"

def get_pipeline_stats():
    """Get pipeline statistics."""
    try:
        stats = pipeline.get_stats()
        
        output = "📊 **Pipeline Statistics:**\n\n"
        output += f"• Processed items: {stats['processed_count']}\n"
        
        cache_stats = stats['cache_stats']
        if cache_stats['enabled']:
            output += f"• Cache entries: {cache_stats.get('entries', 'N/A')}\n"
            if 'oldest' in cache_stats and cache_stats['oldest']:
                output += f"• Oldest cache entry: {cache_stats['oldest']}\n"
        else:
            output += "• Cache: Disabled\n"
        
        return output
        
    except Exception as e:
        return f"❌ Error getting stats: {e}"

def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(
        title="Character Attribute Extractor",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        """
    ) as interface:
        
        gr.Markdown("""
        # 🎨 Character Attribute Extractor
        
        Extract structured attributes from character images or descriptions using AI models.
        """)
        
        with gr.Tabs():
            # Image Processing Tab
            with gr.Tab("📷 Image Processing"):
                gr.Markdown("Upload a character image to extract visual attributes.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            type="pil",
                            label="Character Image",
                            height=400
                        )
                        image_btn = gr.Button(
                            "🔍 Extract Attributes",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        image_output = gr.Markdown(
                            label="Results",
                            value="Upload an image and click 'Extract Attributes' to see results."
                        )
            
            # Text Processing Tab
            with gr.Tab("📝 Text Processing"):
                gr.Markdown("Enter a character description to extract attributes from text.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        text_input = gr.Textbox(
                            label="Character Description",
                            placeholder="e.g., 'A young woman with long black hair and blue eyes wearing a red dress'",
                            lines=5
                        )
                        text_btn = gr.Button(
                            "🔍 Extract Attributes",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        text_output = gr.Markdown(
                            label="Results",
                            value="Enter a description and click 'Extract Attributes' to see results."
                        )
                
                # Example texts
                gr.Markdown("### 💡 Try these examples:")
                gr.Examples(
                    examples=[
                        ["A teenage boy with spiky blonde hair and green eyes"],
                        ["An elderly woman with gray hair in a bun wearing formal attire"],
                        ["A young adult with curly red hair and brown eyes in casual clothes"],
                    ],
                    inputs=text_input,
                    label="Example Descriptions"
                )
            
            # Statistics Tab
            with gr.Tab("📊 Statistics"):
                gr.Markdown("View pipeline performance and cache statistics.")
                
                stats_btn = gr.Button("🔄 Refresh Stats", variant="secondary")
                stats_output = gr.Markdown("Click 'Refresh Stats' to view statistics.")
            
            # About Tab
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## About This Pipeline
                
                This character attribute extraction pipeline uses state-of-the-art AI models to analyze images and text:
                
                ### 🎯 Extracted Attributes
                - **Age**: child, teen, young adult, middle-aged, elderly
                - **Gender**: male, female, non-binary
                - **Ethnicity**: Asian, African, Caucasian, Hispanic, Middle Eastern
                - **Hair Style**: straight, curly, wavy, ponytail, bun, braided
                - **Hair Color**: black, brown, blonde, red, white, gray, colorful
                - **Hair Length**: bald, short, medium, long
                - **Eye Color**: brown, blue, green, gray, hazel, amber
                - **Body Type**: slim, average, muscular, curvy, heavy
                - **Dress**: casual, formal, traditional, uniform, fantasy, modern
                
                ### 🔧 Technology Stack
                - **Image Processing**: CLIP (Contrastive Language-Image Pre-training)
                - **Text Processing**: BART (Bidirectional Auto-Regressive Transformers)
                - **Caching**: SQLite for fast repeated queries
                - **Interface**: Gradio for easy web-based interaction
                
                ### 📈 Scalability Features
                - Batch processing for multiple items
                - Intelligent caching system
                - Parallel processing support
                - Memory-efficient streaming
                - Modular architecture for easy extension
                
                ### 🚀 Production Ready
                This pipeline can scale to process millions of samples with:
                - GPU acceleration
                - Distributed processing
                - Incremental result saving
                - Error recovery mechanisms
                """)
        
        # Event handlers
        image_btn.click(
            fn=process_image,
            inputs=[image_input],
            outputs=[image_output]
        )
        
        text_btn.click(
            fn=process_text,
            inputs=[text_input],
            outputs=[text_output]
        )
        
        stats_btn.click(
            fn=get_pipeline_stats,
            inputs=[],
            outputs=[stats_output]
        )
    
    return interface

def main():
    """Main entry point."""
    try:
        # Initialize pipeline
        initialize_pipeline()
        
        # Create and launch interface
        interface = create_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            debug=False
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

if __name__ == "__main__":
    main()