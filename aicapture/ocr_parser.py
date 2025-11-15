import json
import os
from typing import Any, Dict, List, Optional

from loguru import logger
from PIL import Image

from aicapture.settings import MAX_CONCURRENT_TASKS, ImageQuality
from aicapture.vision_models import VisionModel
from aicapture.vision_parser import DEFAULT_PROMPT, VisionParser

DEFAULT_BLOCK_IDENTIFICATION_PROMPT = """
    Analyze this document image and identify all distinct content blocks.

    For each block, identify its type and provide bounding box coordinates.

    Block types to identify:
    - text: Paragraphs, headings, body text, lists (IMPORTANT: Group lines into regions/paragraphs. Do NOT split individual lines.)
    - table: Tabular data
    - figure: Charts, graphs, plots
    - math_formula: Equations, math formulas
    - diagram: Flowcharts, technical diagrams
    - image: Photographs or illustrations

    Bounding boxes must include some padding, not a tight crop.

    Respond with a JSON array of blocks:
    [
    {
        "type": "text|table|figure|math_formula|diagram|image",
        "bbox": [ymin, xmin, ymax, xmax],  // All values in 0-1000, with padding
    }
    ]

    Return ONLY the JSON array, nothing else.
"""

DEFAULT_BATCH_BLOCK_EXTRACTION_PROMPT = """
    Extract each content block SEPARATELY.

    {user_prompt}

    Blocks (each with its own bbox <|det|>[[ymin, xmin, ymax, xmax]]<|/det|>):
    {blocks_info}

    For each block:
    - Focus on the region inside the block's unique bounding box and include a reasonable margin around it, as sometimes the bbox may not be perfectly accurate.
    - Do NOT duplicate or merge content between blocks.
    - Ignore content clearly outside the bbox and its immediate surroundings.
    - If empty, reply "Block appears empty".

    Strict format for each block:
    <|ref|>{{block_type}}<|/ref|><|det|>[[{{ymin}}, {{xmin}}, {{ymax}}, {{xmax}}]]<|/det|>
    {{extracted_content}}

    Blocks ordered by position (top to bottom; then left to right).

    Block extraction guidance:
    - text: Preserve formatting and reading order.
    - table: Markdown table, all values and headers present.
    - figure: Describe type, axes, major labels, visible trends.
    - math_formula: LaTeX notation for visible equations.
    - diagram: Describe type, label main components and relationships visible.
    - image: Visual description, visible labels/text present inside bbox.
"""


class OCRParser(VisionParser):
    """Parses OCR blocks using block detection and LLM-guided region extraction."""

    def __init__(
        self,
        vision_model: Optional[VisionModel] = None,
        cache_dir: Optional[str] = "./tmp/.orc_cache",
        max_concurrent_tasks: int = MAX_CONCURRENT_TASKS,
        image_quality: str = ImageQuality.DEFAULT,
        invalidate_cache: bool = False,
        cloud_bucket: Optional[str] = None,
        prompt: str = DEFAULT_PROMPT,
        dpi: int = int(os.getenv("VISION_PARSER_DPI", "333")),
        block_prompts: Optional[Dict[str, str]] = None,
        block_identification_prompt: Optional[str] = None,
        use_bounding_box: bool = True,
        use_confidence_score: bool = True,
    ):
        super().__init__(
            vision_model=vision_model,
            cache_dir=cache_dir,
            max_concurrent_tasks=max_concurrent_tasks,
            image_quality=image_quality,
            invalidate_cache=invalidate_cache,
            cloud_bucket=cloud_bucket,
            prompt=prompt,
            dpi=dpi,
        )
        self.block_prompts = block_prompts or {}
        self.block_identification_prompt = (
            block_identification_prompt or DEFAULT_BLOCK_IDENTIFICATION_PROMPT
        )
        self.use_bounding_box = use_bounding_box
        self.use_confidence_score = use_confidence_score

    def _get_block_prompt(self, block_type: str, bbox: List[int]) -> str:
        """Return a concise extraction prompt guiding LLM to extract a region by bbox."""
        if block_type in self.block_prompts:
            prompt_template = self.block_prompts[block_type]
        else:
            _D = {
                "text": "Extract ONLY text inside and reasonably around bbox [ymin, xmin, ymax, xmax]: {bbox}. Exclude irrelevant outside content. Preserve structure and formatting.",
                "table": "Extract table ONLY inside and reasonably around bbox [ymin, xmin, ymax, xmax]: {bbox}. Output as markdown table. No info outside relevant region.",
                "figure": "Describe figure/chart ONLY inside and reasonably around bbox [ymin, xmin, ymax, xmax]: {bbox}. List axis labels, major trends, legends, visible data.",
                "math_formula": "Extract all visible math formulas inside and reasonably around bbox [ymin, xmin, ymax, xmax]: {bbox}, as LaTeX if possible.",
                "diagram": "Describe diagram ONLY inside and reasonably around bbox [ymin, xmin, ymax, xmax]: {bbox}. List visible parts/labels and relationships.",
                "image": "Describe photograph/visual ONLY inside and reasonably around bbox [ymin, xmin, ymax, xmax]: {bbox}. List any text, features, and visible content.",
            }
            prompt_template = _D.get(block_type, self.prompt)
        return prompt_template.format(bbox=bbox)

    def _get_batch_extraction_prompt(
        self, blocks: List[Dict[str, Any]], image_width: int, image_height: int
    ) -> str:
        """Build batch extraction prompt listing all blocks as pixel bboxes."""
        sorted_blocks = sorted(blocks, key=lambda b: (b["bbox"][0], b["bbox"][1]))

        def norm2pix(v, dim):
            return int(v / 1000.0 * dim)

        lines = []
        for i, b in enumerate(sorted_blocks, 1):
            ymin, xmin, ymax, xmax = b["bbox"]
            lines.append(
                f"Block {i}: type={b['type']}, bbox_pixels=[ymin={norm2pix(ymin, image_height)}, xmin={norm2pix(xmin, image_width)}, ymax={norm2pix(ymax, image_height)}, xmax={norm2pix(xmax, image_width)}]"
            )
        prompt = DEFAULT_BATCH_BLOCK_EXTRACTION_PROMPT.format(
            user_prompt=self.prompt, blocks_info="\n".join(lines)
        )

        if self.use_confidence_score:
            confidence_instruction = """
        
                After extracting the content, provide a VERY conservative confidence assessment at the end of your response using the format below.

                The most important factor for your confident_score is the condition of the document. Start by carefully evaluating visible document quality: look for any blurriness, poor scans, handwriting, faded text, low resolution, heavy marks, or any other visual degradations. If the document is less than perfectly clear, your confidence should be substantially lower, and you must briefly note these specific condition issues in the confident_reason.
                Next, consider your assessment of the accuracy of the extracted content, especially for all numbers (amounts, account numbers, dates, or any numeric values). If there are errors, ambiguities, or information is unclear, reduce your confident_score accordingly and briefly describe the issues in the confident_reason.

                The confidence assessment must follow this structure:

                <confidence_assessment>
                {
                "confident_score": <integer between 0 and 100>,
                "confident_reason": "<brief but clear explanation for the score, noting the document condition (such as scan quality or legibility) first, and any uncertainties or ambiguities in the extracted content (especially numbers). For example: 'Document is high quality, all numbers are clear', or 'Amount field ambiguous due to faded and blurry area on scan'>"
                }
                </confidence_assessment>

                The confident_score should be conservative, focusing first on the visible quality and condition of the document, and then on the accuracy of the extracted content. Only assign a score near 100 if the document condition is excellent and there are no extraction errors or issues. Make the confident_reason detailed but concise.
            """
            prompt = prompt + confidence_instruction

        return prompt

    async def _identify_blocks_async(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Identify all document blocks with normalized bboxes."""
        try:
            logger.debug("Identifying document blocks...")
            prompt = self.block_identification_prompt
            async with self.__class__._semaphore:
                response = await self.vision_model.process_image_async(
                    image, prompt=prompt
                )
            response = response.strip()
            json_start, json_end = response.find("["), response.rfind("]") + 1
            if json_start == -1 or json_end == 0:
                logger.warning("No JSON array found in block identification response")
                return []
            blocks = json.loads(response[json_start:json_end])
            valid_types = {
                "text",
                "table",
                "figure",
                "math_formula",
                "diagram",
                "image",
            }
            validated = []
            for block in blocks:
                t = (block.get("type") or "").lower()
                bbox = block.get("bbox", [])
                if (
                    t not in valid_types
                    or not (
                        isinstance(bbox, list)
                        and len(bbox) == 4
                        and all(isinstance(x, (int, float)) for x in bbox)
                    )
                    or any(not (0 <= x <= 1000) for x in bbox)
                    or bbox[0] >= bbox[2]
                    or bbox[1] >= bbox[3]
                ):
                    logger.warning(f"Invalid block: {block}")
                    continue
                validated.append({"type": t, "bbox": bbox})
            logger.info(f"Identified {len(validated)} document blocks")
            return validated
        except Exception as e:
            logger.error(f"Error identifying blocks: {e}")
            return []

    async def _process_block_async(
        self, image: Image.Image, block_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract a block using full image, guiding LLM to focus on bbox."""
        try:
            block_type, bbox = block_info["type"], block_info["bbox"]
            prompt = self._get_block_prompt(block_type, bbox)
            async with self.__class__._semaphore:
                content = await self.vision_model.process_image_async(
                    image, prompt=prompt
                )
            cleaned = self.content_cleaner.clean_content(content.strip())
            return {
                "type": block_type,
                "bbox": bbox,
                "content": cleaned,
            }
        except Exception as e:
            logger.error(
                f"Error processing block {block_info.get('type', 'unknown')}: {e}"
            )
            return {
                "type": block_info["type"],
                "bbox": block_info["bbox"],
                "content": f"[Error extracting content: {str(e)}]",
            }

    async def process_page_async(
        self,
        image: Image.Image,
        page_number: int,
        page_hash: str,
        text_content: str = "",
    ) -> Dict:
        """Batch-mode: process all blocks in a single LLM call."""
        if not self.use_bounding_box:
            return await super().process_page_async(
                image, page_number, page_hash, text_content
            )

        try:
            logger.debug(f"Processing page {page_number} with ORCParser (batch mode)")
            blocks = await self._identify_blocks_async(image)
            if not blocks:
                logger.warning(
                    f"No blocks for page {page_number}, fallback to standard processing"
                )
                return await super().process_page_async(
                    image, page_number, page_hash, text_content
                )

            image_width, image_height = image.size
            prompt = self._get_batch_extraction_prompt(
                blocks, image_width, image_height
            )
            logger.debug(f"Batch extraction prompt: {prompt}")

            async with self.__class__._semaphore:
                response = await self.vision_model.process_image_async(
                    image, prompt=prompt
                )
            logger.debug(f"Batch response: {response}")

            if self.use_confidence_score:
                # Parse confidence from response
                content_without_confidence, confident_score, confident_reason = (
                    self._parse_confidence_from_response(response)
                )
                # Clean the content
                cleaned_content = self.content_cleaner.clean_content(
                    content_without_confidence.strip()
                )
                return {
                    "page_number": page_number,
                    "page_content": cleaned_content,
                    "page_hash": page_hash,
                    "confident_score": confident_score,
                    "confident_reason": confident_reason,
                }
            else:
                # Use default values when confidence assessment is disabled
                cleaned_content = self.content_cleaner.clean_content(response.strip())
                return {
                    "page_number": page_number,
                    "page_content": cleaned_content,
                    "page_hash": page_hash,
                    "confident_score": 30,
                    "confident_reason": "Batch mode",
                }

        except Exception as e:
            logger.error(
                f"Error processing page {page_number} with ORCParser (batch mode): {e}"
            )
            return await super().process_page_async(
                image, page_number, page_hash, text_content
            )
