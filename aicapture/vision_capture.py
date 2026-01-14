"""
Vision Capture module for extracting structured data from documents using templates.

This module provides the ``VisionCapture`` class, which combines document parsing
(via VisionParser) with Vision Language Model (VLM) capabilities to extract
structured information from PDFs and images based on user-defined templates.

Key Features:
    - Template-based data extraction from PDFs and images
    - Support for multiple VLM providers (OpenAI, Anthropic, Gemini, Azure, Bedrock)
    - Automatic document parsing with OCR capabilities
    - Flexible template format for defining expected output structure

Typical Usage:
    >>> import asyncio
    >>> from aicapture import VisionCapture
    >>>
    >>> async def extract_invoice_data():
    ...     capture = VisionCapture()
    ...     template = '''
    ...     invoice_number: <extract invoice number>
    ...     date: <extract invoice date in YYYY-MM-DD format>
    ...     total_amount: <extract total amount as number>
    ...     vendor_name: <extract vendor/company name>
    ...     '''
    ...     result = await capture.capture("invoice.pdf", template)
    ...     return result
    >>>
    >>> # Run the async function
    >>> data = asyncio.run(extract_invoice_data())

Template Format:
    Templates are plain text strings that define the expected structure of the
    output. The VLM will extract information from the document and fill in
    the template fields. Templates can be:

    1. Simple key-value format::

        field_name: <description of what to extract>
        another_field: <description>

    2. JSON-like structure::

        {
            "field1": "<extract description>",
            "field2": "<extract description>",
            "nested": {
                "subfield": "<extract description>"
            }
        }

    3. YAML-like format::

        section:
          field1: <extract description>
          field2: <extract description>
        another_section:
          field3: <extract description>

    The VLM interprets the template and returns structured output matching
    the requested format.

See Also:
    - :class:`aicapture.VisionParser`: For lower-level document parsing
    - :class:`aicapture.VisionModel`: For direct VLM interactions
    - :class:`aicapture.VidCapture`: For video frame extraction and analysis
"""

from typing import Any, Dict, Optional

from aicapture.vision_models import VisionModel, create_default_vision_model
from aicapture.vision_parser import VisionParser


class VisionCapture:
    """
    Extract structured data from documents using Vision Language Models and templates.

    VisionCapture provides a high-level interface for extracting specific information
    from PDFs and images. It combines document parsing (converting documents to text
    via OCR) with VLM-based information extraction using user-defined templates.

    The extraction process works in two stages:
        1. **Document Parsing**: The input file (PDF or image) is processed by
           VisionParser, which uses a VLM to extract text content from each page.
        2. **Template Extraction**: The extracted text is then processed by the
           VLM again, using the provided template to structure the output.

    Attributes:
        vision_model (VisionModel): The Vision Language Model used for processing.
            Defaults to auto-detected model based on available API keys.
        vision_parser (VisionParser): The parser used for initial document processing.
            Created automatically if not provided.

    Example:
        Basic usage with default settings::

            import asyncio
            from aicapture import VisionCapture

            async def main():
                capture = VisionCapture()

                # Define what data to extract
                template = '''
                patient_name: <full name of patient>
                date_of_birth: <DOB in MM/DD/YYYY format>
                diagnosis: <primary diagnosis>
                medications: <list of prescribed medications>
                '''

                result = await capture.capture("medical_form.pdf", template)
                print(result)

            asyncio.run(main())

        Using a specific vision model::

            from aicapture import VisionCapture
            from aicapture.vision_models import AnthropicVisionModel

            # Use Claude for extraction
            model = AnthropicVisionModel(model="claude-sonnet-4-5-20250929")
            capture = VisionCapture(vision_model=model)

        Extracting data from an image::

            async def extract_from_image():
                capture = VisionCapture()
                template = '''
                {
                    "product_name": "<name of the product>",
                    "price": "<price as a number>",
                    "description": "<brief product description>"
                }
                '''
                result = await capture.capture("product_label.jpg", template)
                return result

    Supported File Types:
        - PDF files (.pdf)
        - Images: JPG, JPEG, PNG, TIFF, WebP, BMP

    Note:
        - Processing time depends on document size and VLM provider
        - Results are cached by VisionParser to avoid redundant API calls
        - For best results, use clear, specific descriptions in templates
        - The template format is flexible; the VLM adapts to various structures

    See Also:
        - :meth:`capture`: Main method for extracting structured data
        - :class:`VisionParser`: Lower-level document parsing
        - :func:`create_default_vision_model`: Auto-detection of VLM providers
    """

    def __init__(
        self,
        vision_model: Optional[VisionModel] = None,
        vision_parser: Optional[VisionParser] = None,
    ) -> None:
        """
        Initialize VisionCapture with optional custom model and parser.

        Args:
            vision_model: Custom VisionModel instance for VLM processing.
                If not provided, automatically creates one based on available
                API keys (checks Gemini, OpenAI, Azure, Anthropic in order).
                See :func:`create_default_vision_model` for auto-detection logic.
            vision_parser: Custom VisionParser instance for document processing.
                If not provided, creates a new VisionParser using the vision_model.
                Pass a custom parser to configure caching, DPI settings, or
                other parsing options.

        Example:
            Default initialization (auto-detect provider)::

                capture = VisionCapture()

            With custom model::

                from aicapture.vision_models import OpenAIVisionModel

                model = OpenAIVisionModel(
                    model="gpt-4o",
                    api_key="your-api-key"
                )
                capture = VisionCapture(vision_model=model)

            With custom parser (e.g., to disable caching)::

                from aicapture import VisionParser

                parser = VisionParser(invalidate_cache=True)
                capture = VisionCapture(vision_parser=parser)

        Raises:
            ValueError: If no API key is found for any supported VLM provider
                when using auto-detection.

        Note:
            When both vision_model and vision_parser are provided, the
            vision_model parameter is ignored (parser's model is used).
        """
        self.vision_model = vision_model or create_default_vision_model()
        self.vision_parser = vision_parser or VisionParser(vision_model=self.vision_model)

    async def _parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a document file and extract its content using VisionParser.

        This internal method routes files to the appropriate parsing method
        based on file extension. PDFs are processed page-by-page, while
        images are processed as single-page documents.

        Args:
            file_path: Absolute or relative path to the document file.
                Must be a PDF or supported image format (JPG, JPEG, PNG,
                TIFF, WebP, BMP).

        Returns:
            Dictionary containing the parsed document structure::

                {
                    "file_object": {
                        "pages": [
                            {
                                "page_number": 1,
                                "page_content": "extracted text content...",
                                "word_count": 150
                            },
                            # ... more pages for PDFs
                        ]
                    }
                }

        Raises:
            ValueError: If the file type is not supported (not PDF or
                recognized image format).
            FileNotFoundError: If the specified file does not exist.
            Exception: Various exceptions from VisionParser if document
                processing fails (e.g., corrupted file, API errors).

        Note:
            Results are cached by VisionParser. Subsequent calls with the
            same file will return cached results unless invalidate_cache
            is set on the parser.
        """
        if file_path.endswith(".pdf"):
            return await self.vision_parser.process_pdf_async(file_path)
        elif file_path.endswith(tuple(self.vision_parser.SUPPORTED_IMAGE_FORMATS)):
            return await self.vision_parser.process_image_async(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    async def capture(self, file_path: str, template: str) -> str:
        """
        Extract structured data from a document using a template.

        This is the main method for extracting information from documents.
        It parses the document to extract text content, then uses a Vision
        Language Model to extract specific fields based on the provided template.

        Args:
            file_path: Path to the document file (PDF or image).
                Can be absolute or relative path. Supported formats:
                - PDF files (.pdf)
                - Images: .jpg, .jpeg, .png, .tiff, .webp, .bmp
            template: A string defining the structure of data to extract.
                The template describes what information to extract and in
                what format. The VLM interprets the template and fills in
                the values from the document content.

        Returns:
            String containing the extracted data in the format specified
            by the template. The exact format depends on your template
            structure (plain text, JSON, YAML, etc.).

        Raises:
            ValueError: If the file type is not supported.
            FileNotFoundError: If the specified file does not exist.
            Exception: Various exceptions from underlying VLM or parser
                if processing fails.

        Example:
            Extract invoice information::

                template = '''
                invoice_number: <the invoice number>
                date: <invoice date in YYYY-MM-DD format>
                vendor: <company name>
                line_items:
                  - description: <item description>
                    quantity: <quantity>
                    unit_price: <price per unit>
                    total: <line total>
                subtotal: <subtotal before tax>
                tax: <tax amount>
                total: <grand total>
                '''
                result = await capture.capture("invoice.pdf", template)

            Extract data as JSON::

                template = '''
                {
                    "form_type": "<type of form>",
                    "submitted_by": "<name of submitter>",
                    "submission_date": "<date in ISO format>",
                    "fields": {
                        "field1": "<value>",
                        "field2": "<value>"
                    }
                }
                '''
                result = await capture.capture("form.pdf", template)
                # result is a JSON-formatted string

            Extract from an image::

                template = '''
                product: <product name>
                brand: <brand name>
                price: <price as number only>
                expiry_date: <expiration date if visible, else "N/A">
                '''
                result = await capture.capture("product_photo.jpg", template)

        Template Tips:
            - Be specific about the format you want (e.g., "in YYYY-MM-DD format")
            - Use "N/A" or similar defaults for optional fields
            - Include context in descriptions (e.g., "company name from header")
            - For lists, show the expected structure with examples
            - JSON templates return JSON-formatted strings that can be parsed

        Performance:
            - First call parses document (may be slow for large PDFs)
            - Subsequent calls with same file use cached parsing results
            - Each unique template requires a new VLM API call

        See Also:
            - :meth:`_capture_content`: Lower-level content extraction
            - :class:`VisionParser`: For direct document parsing
        """
        doc_json = await self._parse_file(file_path)
        # Extract content from file_object structure
        content = "\n".join(page["page_content"] for page in doc_json["file_object"]["pages"])
        return await self._capture_content(content, template)

    async def _capture_content(self, content: str, template: str) -> str:
        """
        Extract structured data from text content using a template.

        This internal method sends the document content and template to the
        Vision Language Model for extraction. It constructs a prompt that
        instructs the VLM to extract information matching the template format.

        Args:
            content: The text content extracted from the document.
                This is typically the concatenated page content from
                VisionParser's output.
            template: Template string defining the extraction structure.
                See :meth:`capture` for template format documentation.

        Returns:
            String containing the VLM's response with extracted data
            structured according to the template.

        Raises:
            Exception: Various exceptions from the VLM provider if the
                API call fails (rate limits, authentication, etc.).

        Note:
            This method is called internally by :meth:`capture`. Use
            :meth:`capture` for the standard document extraction workflow.
            Only call this method directly if you have pre-extracted
            document content.

        Internal Details:
            The prompt sent to the VLM follows this structure::

                Extract information and output in the template format:
                <template>{template}</template>
                Content:
                <content>{content}</content>

            The VLM interprets this prompt and returns structured output
            matching the template format.
        """
        message = (
            f"Extract information and output in the template format: \n"
            f"<template>{template}</template>\n"
            f"Content: \n"
            f"<content>{content}</content>"
        )
        messages = [
            {"role": "user", "content": message},
        ]
        return await self.vision_model.process_text_async(messages)
