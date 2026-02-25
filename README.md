<p align="center">
  <img src="static/logo.svg" width="110" alt="OpenBanana Logo"/>
</p>

<h1 align="center">OpenBanana</h1>
<p align="center"><em>Turn any diagram image into a fully editable DrawIO file — runs entirely on your machine</em></p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-2F80ED?style=flat-square&logo=apache&logoColor=white" alt="License"/></a>
  <a href="https://developer.apple.com/metal/"><img src="https://img.shields.io/badge/Apple_M--series-MPS_Ready-black?style=flat-square&logo=apple&logoColor=white" alt="Apple Silicon"/></a>
  <a href="https://developer.nvidia.com/cuda-downloads"><img src="https://img.shields.io/badge/NVIDIA-CUDA_Supported-76B900?style=flat-square&logo=nvidia" alt="CUDA"/></a>
</p>

---

OpenBanana takes a screenshot, scan, or export of any diagram — flowchart, system architecture, UML, network map — and reconstructs it as a `.drawio` file. Every shape, arrow, and label becomes a real, independently editable element. No cloud, no API keys, no subscriptions.

## How it works

```
Image or PDF
  ↓  Downscale if either dimension exceeds 1500 px
  ↓  SAM3 segments the diagram into shapes, arrows, icons, and background regions
  ↓  Each segment is cropped and sent to a local vision model for text extraction
  ↓  Fill color, stroke, and style are extracted per element
  ↓  Arrow endpoints and direction are detected
  ↓  All elements are assembled into layered DrawIO XML
.drawio file
```

## What runs locally

OpenBanana is designed to run entirely on your own hardware:

- **SAM3 segmentation** runs as a local FastAPI service on port 8001. It uses your GPU (CUDA) or Apple Neural Engine / MPS on Apple Silicon Macs (M1 through M3).
- **Text labeling** is handled by a local vision LLM served through [LM Studio](https://lmstudio.ai/) on port 1234. Any OpenAI-compatible local server works — LM Studio, Ollama, llama.cpp with an HTTP frontend, etc. We use `allenai/olmocr-2-7b` which is optimized for OCR tasks.
- **No data leaves your machine.** Nothing is sent to external APIs unless you explicitly configure a remote model endpoint.

## Running on Apple Silicon (M1 / M2 / M3)

OpenBanana works well on Apple M-series Macs. A few things to know:

SAM3 uses some PyTorch ops that are not yet implemented for MPS. The `PYTORCH_ENABLE_MPS_FALLBACK=1` environment variable tells PyTorch to fall those specific ops back to CPU while keeping everything else on the Metal backend. In practice this means you get near-native speed for most of the segmentation work.

```bash
# Always prefix server and CLI commands with this on Apple Silicon:
PYTORCH_ENABLE_MPS_FALLBACK=1 python server_pa.py
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py -i input/diagram.png
```

Also: torch 2.6.0 requires torchvision 0.21.0 specifically on macOS. The `pip install torchvision` default will install a newer incompatible version.

```bash
pip install torchvision==0.21.0
```

## Setup

### 1. Clone and install

```bash
git clone https://github.com/Kaleemullahqasim/openbanana.git
cd openbanana
pip install -r requirements.txt
pip install torchvision==0.21.0   # macOS / torch 2.6.0
```

Install PyTorch separately for your hardware: [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)

### 2. Download SAM3 weights

| File | Source | Place at |
|------|--------|----------|
| `sam3.pt` | [ModelScope — facebook/sam3](https://modelscope.cn/models/facebook/sam3) | `models/sam3/sam3.pt` |
| `bpe_simple_vocab_16e6.txt.gz` | Bundled in the same ModelScope download | `models/bpe_simple_vocab_16e6.txt.gz` |

### 3. Configure

```bash
cp config/config.yaml.example config/config.yaml
```

Minimum required edits in `config/config.yaml`:

```yaml
sam3:
  checkpoint_path: "models/sam3/sam3.pt"
  bpe_path: "models/bpe_simple_vocab_16e6.txt.gz"

multimodal:
  local_base_url: "http://localhost:1234/v1"   # LM Studio default
  local_api_key:  "lm-studio"
  local_model:    "allenai/olmocr-2-7b"        # or any vision model you have loaded
```

To use Ollama instead of LM Studio:

```yaml
multimodal:
  local_base_url: "http://localhost:11434/v1"
  local_api_key:  "ollama"
  local_model:    "llava"   # or whichever multimodal model you have pulled
```

### 4. Set up local LLM (for text extraction)

Text inside shapes is extracted by a local vision model. Without it the pipeline still runs — shapes will just have no labels.

**LM Studio:**
1. Download [LM Studio](https://lmstudio.ai/) and install `allenai/olmocr-2-7b`
2. Go to **Developer → Start Server** (starts on port 1234)

**Ollama:**
```bash
ollama pull llava
ollama serve
```

## Running

### Start the SAM3 service

The segmentation model runs as a separate process. Start it before the web server or CLI:

```bash
# Apple Silicon
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m sam3_service.server --port 8001 --device mps

# CUDA
python -m sam3_service.server --port 8001 --device cuda
```

### Web UI

```bash
# Apple Silicon
PYTORCH_ENABLE_MPS_FALLBACK=1 python server_pa.py

# CUDA / CPU
python server_pa.py
```

Open [http://localhost:8000](http://localhost:8000), upload an image or PDF, and download the `.drawio` result. The UI shows live progress through each pipeline stage.

### CLI

```bash
# Single image
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py -i input/diagram.png

# All images in input/
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py

# Skip text extraction (faster, no LLM needed)
python main.py -i input/diagram.png --no-text

# Process only specific element types
python main.py -i input/diagram.png --groups shape arrow

# Run a refinement pass after the initial output
python main.py -i input/diagram.png --refine
```

Output is saved to `output/<filename>/`.

## Architecture

All processors share a `ProcessingContext` that is passed through the pipeline in `main.py`:

```
Input
  ↓ TextRestorer          Full-image OCR pass (optional, Azure or local VLM)
  ↓ Sam3InfoExtractor     Calls the SAM3 service; builds a list of ElementInfo objects
  ↓ IconPictureProcessor  Encodes icon/image segments as base64
  ↓ BasicShapeProcessor   Extracts fill color and stroke; produces DrawIO style strings
  ↓ ArrowProcessor        Determines arrow direction and connects endpoints
  ↓ XMLMerger             Sorts elements by layer and writes the final .drawio file
Output
```

Layer order (bottom to top): `BACKGROUND → BASIC_SHAPE → IMAGE → ARROW → TEXT`

Key files:

| File | What it does |
|------|-------------|
| `modules/data_types.py` | Shared types: `ElementInfo`, `ProcessingContext`, `LayerLevel` |
| `modules/sam3_info_extractor.py` | SAM3 service client; returns segmented elements |
| `modules/vlm_labeler.py` | Crops each shape and queries the local vision model |
| `modules/xml_merger.py` | Assembles the final DrawIO XML document |
| `sam3_service/server.py` | FastAPI wrapper around the SAM3 model |
| `config/config.yaml` | All runtime settings |

## Roadmap

| Feature | Status |
|---------|--------|
| SAM3 segmentation → DrawIO export | ✅ Done |
| Local VLM text labeling (LM Studio / Ollama) | ✅ Done |
| Apple Silicon (MPS) support | ✅ Done |
| Auto-downscale large images | ✅ Done |
| Live pipeline progress in web UI | ✅ Done |
| Segment confidence cap (top 80) | ✅ Done |
| Smart arrow-to-shape connection | 🔄 In progress |
| PPTX export | 📍 Planned |
| Batch PDF processing | 📍 Planned |

## Contributing

Open an issue first for anything non-trivial. For small fixes, a PR is fine directly.

```bash
git checkout -b feature/your-feature
git commit -m 'feat: what you changed and why'
git push origin feature/your-feature
# open a pull request
```

## Acknowledgements

OpenBanana was inspired by the Edit Banana project from the BIT DataLab research group, which first demonstrated the idea of using SAM-based segmentation to reconstruct diagrams as editable files. The fine-tuned SAM3 weights used here come from that work. OpenBanana takes that foundation in a new direction: a fully local, open-source tool anyone can run on their own machine without cloud dependencies.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
