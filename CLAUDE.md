# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OpenBanana** converts static diagram images (PNG/JPG/PDF) into editable DrawIO XML (`.drawio`) or PPTX files. The pipeline uses a fine-tuned SAM3 (Segment Anything Model 3) for element segmentation, combined with multimodal LLMs and Azure OCR for text extraction.

## Setup

1. Copy `config/config.yaml.example` to `config/config.yaml` and fill in:
   - `sam3.checkpoint_path` — path to the SAM3 `.pt` model weights
   - `sam3.bpe_path` — path to `bpe_simple_vocab_16e6.txt.gz`
   - `multimodal.*` — VLM API credentials (Azure, Mistral, or local Ollama)

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   # PyTorch/SAM3 must be installed separately per your CUDA setup
   ```

## Running the System

**On Apple Silicon (MPS), prefix commands with:**
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py ...
```
Some SAM3 ops aren't implemented for MPS; this env var falls them back to CPU automatically.

**CLI (single image or batch):**
```bash
python main.py -i input/test.png          # single image
python main.py                             # batch: all images in input/
python main.py -i input/test.png --refine  # with iterative refinement
python main.py -i input/test.png --no-text # skip OCR
python main.py -i input/test.png --groups image arrow shape  # specific element groups only
python main.py --show-prompts              # show prompt configuration
```

**Web backend (FastAPI):**
```bash
python server_pa.py
# Runs on http://localhost:8000
# POST /convert — upload image/PDF, returns .drawio XML or PPTX
```

**SAM3 inference service (required by the pipeline):**
```bash
# Single GPU
python -m sam3_service.server --port 8001 --device cuda

# Multi-GPU load balancing
CUDA_VISIBLE_DEVICES=0 python -m sam3_service.server --port 8001 --device cuda
CUDA_VISIBLE_DEVICES=1 python -m sam3_service.server --port 8002 --device cuda

# Or use the launcher
python -m sam3_service.run_all_service --workers 2
```

## Architecture

The pipeline is orchestrated in `main.py` using a shared `ProcessingContext` object that flows through each stage:

```
Input Image
  → [TextRestorer]         OCR text + formula (LaTeX) extraction via Azure / VLM fallback
  → [Sam3InfoExtractor]    SAM3 segmentation into 4 prompt groups: image, arrow, shape, background
  → [IconPictureProcessor] Icons/images encoded as base64 (optional upscaling via spandrel)
  → [BasicShapeProcessor]  Color/stroke extraction → DrawIO style strings
  → [ArrowProcessor]       Arrow direction + endpoint detection → edge XML
  → [XMLMerger]            Assemble fragments sorted by LayerLevel → final .drawio file
  → [MetricEvaluator]      (optional) Score reconstruction quality
  → [RefinementProcessor]  (optional) Re-run on low-quality regions
```

### Key files

| File | Role |
|------|------|
| `modules/data_types.py` | Shared data structures: `ElementInfo`, `ProcessingContext`, `ElementType`, `LayerLevel` |
| `modules/base.py` | `BaseProcessor` abstract class all processors inherit from |
| `modules/sam3_info_extractor.py` | Calls SAM3 service; returns list of `ElementInfo` with masks, bboxes, polygons |
| `modules/text/restorer.py` | Orchestrates OCR pipeline (Azure primary, VLM fallback, formula detection) |
| `modules/xml_merger.py` | Merges per-element XML fragments into a DrawIO document |
| `prompts/*.py` | Text prompt definitions per element group (`image`, `arrow`, `shape`, `background`) |
| `sam3_service/server.py` | FastAPI inference service wrapping the SAM3 model |
| `sam3_service/client.py` | Client used by `Sam3InfoExtractor` to call the service |
| `config/config.yaml` | Runtime config (model paths, thresholds, VLM credentials) |

### Layer ordering

Elements are composited in `LayerLevel` order (bottom to top):
`BACKGROUND(0) → BASIC_SHAPE(1) → IMAGE(2) → ARROW(3) → TEXT(4)`

### Adding a new processor

1. Subclass `BaseProcessor` from `modules/base.py`
2. Accept `ProcessingContext` and populate `context.elements` or append to XML fragments
3. Register it in `modules/__init__.py`
4. Add it to the pipeline in `main.py`
