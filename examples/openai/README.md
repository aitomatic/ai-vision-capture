# OpenAI Vision Capture Example

This example demonstrates how to use the Vision Capture feature with OpenAI's Vision model to extract structured data from technical documents.

## Prerequisites

1. Install the required dependencies:
```bash
pip install openai pillow
```

2. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

1. Place your technical document (PDF or image) in the `examples/data` directory.

2. Run the example script:
```bash
python examples/openai/capture_example.py
```

## Example Output

The script will output structured data in YAML format, following this template:

```yaml
alarm:
  description: "INTERVENTO DIFFERIBILE"
  destination: "BCU"
  tag: "ID"
  ref_logica: 83

dependencies:
  - signal_name: "MANCANZA ALIMENTAZIONE D"
    source: "BCU"
    tag: "221D"
    ref_logica: 37
  # ... additional dependencies
```

## Customization

You can modify the `ALARM_TEMPLATE` in the script to match your specific document structure needs. The template uses a YAML-like format to define the expected structure of the extracted data. 