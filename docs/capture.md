# Document Capture Feature

## Overview
The Document Capture feature transforms unstructured documents (PDFs/images) into structured data using a two-stage process:

1. Document Parsing (using Vision Parser)
   - Extracts raw content from documents
   - Preserves document structure and relationships
   - Maintains formatting and hierarchies

2. Template-Based Structuring
   - Maps extracted content to predefined templates
   - Validates data against schema requirements
   - Generates standardized output formats

## Key Features
- Template-driven data extraction
- Schema validation and enforcement
- Support for multiple output formats
- Configurable field mapping
- Data normalization and cleaning

## Template Schema
Templates define:
- Required fields and their types
- Field validation rules
- Data transformation rules
- Output format specifications

## Output Formats
Supports structured outputs in:
- JSON
- YAML

## Example Template
Here's an example of how a technical alarm logic document gets structured:

```yaml
# Alarm Logic Template
alarm:
  description: string
  destination: string
  tag: string
  ref_logica: integer

dependencies:
  type: array
  items:
    - signal_name: string
      source: string
      tag: string
      ref_logica: integer|null

# Example Output
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
  - signal_name: "MANCANZA ALIMENTAZIONE MANOVRA SEZIONATORI"
    source: "P05_BI13"
    tag: "223"
    ref_logica: null
  - signal_name: "INDISPONIBILITA' SEZIONATORI"
    source: "BCU"
    tag: "A89"
    ref_logica: 43
  # ... additional dependencies follow same structure
```

## Usage Flow
1. Process document through Vision Parser
2. Apply template mapping
3. Validate structured output
4. Generate final formatted output