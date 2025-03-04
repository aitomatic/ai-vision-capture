from vision_capture import VisionParser, GeminiVisionModel
from vision_capture.settings import ImageQuality

def main():
    # Initialize Gemini vision model (API key will be loaded from .env)
    vision_model = GeminiVisionModel(
        model="gemini-2.0-flash",
        temperature=0.3,
    )

    # Initialize parser
    parser = VisionParser(
        vision_model=vision_model,
        image_quality=ImageQuality.HIGH,
        cache_dir="./cache",
        prompt="""
        Extract from this technical document:
        1. Main topics and key points
        2. Technical specifications
        3. Important procedures
        4. Tables and data (in markdown)
        5. Diagrams and figures
        """
    )

    # Process a single PDF
    result = parser.process_pdf("path/to/your/document.pdf")
    
    # Save results
    parser.save_output(result, "output.json")
    parser.save_markdown_output(result, "output")
    
    print(f"Document: {result['file_object']['file_name']}")
    print(f"Pages: {result['file_object']['total_pages']}")
    print(f"Words: {result['file_object']['total_words']}")

if __name__ == "__main__":
    main() 