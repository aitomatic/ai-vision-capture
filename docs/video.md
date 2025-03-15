# Video Capture Tool

This document outlines the implementation of a video capture tool that extracts knowledge from video content using our existing vision infrastructure.

## Overview

The video capture tool will leverage our existing vision model infrastructure to process video frames and extract meaningful information. The system will:
1. Process video files to extract frames at configurable intervals
2. Analyze frames using our vision models
3. Combine frame analysis with temporal context
4. Generate structured knowledge output

## Implementation Requirements

### 1. Video Processing Configuration
- Frame extraction rate (frames per second)
- Maximum video duration (default: 30 seconds)
- Maximum number of frames to process
- Video quality settings
- Supported video formats (e.g., MP4, AVI, MOV)

### 2. Frame Processing
- Use existing vision model infrastructure
- Support batch processing of frames
- Maintain temporal context between frames
- Handle high-resolution video frames efficiently

### 3. Knowledge Extraction
- Scene detection and segmentation
- Object tracking across frames
- Action recognition
- Text overlay extraction (OCR)
- Audio transcription integration (optional)

### 4. Output Format
- Structured JSON output
- file name
- file hash
- file size
- extracted knowledge

## Integration with Existing Infrastructure

### Vision Models Integration
```python
from vision_capture.vision_models import VisionModel
from vision_capture.vision_parser import VisionParser
from vision_capture.vision_capture import VisionCapture

# Example configuration for video processing
VIDEO_CONFIG = {
    "max_duration_seconds": 30,
    "frame_rate": 1,  # Extract 1 frame per second
    "max_frames": 30,
    "batch_size": 5,  # Process 5 frames at a time
    "min_confidence": 0.7
}
```

### Frame Processing Pipeline
```python
import cv2
import base64
from typing import List, Dict

async def process_video_frames(
    video_path: str,
    vision_model: VisionModel,
    config: Dict
) -> List[Dict]:
    """
    Process video frames using the vision model.
    
    Args:
        video_path: Path to video file
        vision_model: Instance of VisionModel
        config: Video processing configuration
        
    Returns:
        List of frame analysis results
    """
    video = cv2.VideoCapture(video_path)
    frame_results = []
    
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
            
        # Convert frame to base64
        _, buffer = cv2.imencode(".jpg", frame)
        base64_frame = base64.b64encode(buffer).decode("utf-8")
        
        # Process frame with vision model
        result = await vision_model.process_image(frame)
        frame_results.append(result)
    
    video.release()
    return frame_results
```

The capture should process multiple frames at once, and return the result .
Example bare python code:
```python
# OpenAI example
video = cv2.VideoCapture("data/bison.mp4")

base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()
print(len(base64Frames), "frames read.")

PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "These are frames from a video that I want to upload. Generate a compelling description that I can upload along with the video.",
            *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::50]),
        ],
    },
]
params = {
    "model": "gpt-4o",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 200,
}

result = client.chat.completions.create(**params)
print(result.choices[0].message.content)
```

## Implementation Steps

1. Create a new `VidCapture` class extending from base `VisionParser`:
   - Add video-specific configuration
   - Implement frame extraction logic
   - Add batch processing support
   - Handle temporal context

2. Add video-specific prompts and templates:
   - Scene description templates
   - Action recognition prompts
   - Object tracking templates
   - Text extraction prompts


## Usage Example

```python
from vision_capture import VisionCapture, VideoConfig

# Initialize video capture
video_capture = VisionCapture()

# Process video file
result = await video_capture.capture(
    file_path="example.mp4",
    template="video_analysis",
    config=VideoConfig(
        max_duration=30,
        frame_rate=1,
        batch_size=5
    )
)
```
