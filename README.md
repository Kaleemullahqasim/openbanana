<p align="center">
  <img src="static/logo.svg" width="110" alt="OpenBanana Logo"/>
</p>

<h1 align="center">OpenBanana</h1>
<p align="center"><em>Turn any diagram image into a fully editable DrawIO file</em></p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-2F80ED?style=flat-square&logo=apache&logoColor=white" alt="License"/></a>
  <a href="https://developer.apple.com/metal/"><img src="https://img.shields.io/badge/Apple_Silicon-MPS_Supported-black?style=flat-square&logo=apple&logoColor=white" alt="MPS"/></a>
  <a href="https://developer.nvidia.com/cuda-downloads"><img src="https://img.shields.io/badge/GPU-CUDA_Supported-76B900?style=flat-square&logo=nvidia" alt="CUDA"/></a>
</p>

---

Upload a PNG, JPG, or PDF of a flowchart, architecture diagram, UML diagram, or any technical drawing. OpenBanana segments every element — shapes, arrows, icons, and text — and outputs a `.drawio` file where each one is independently selectable, movable, and editable.

## How it works

```
Input (image or PDF)
  ↓  Auto-resize if wider/taller than 1500 px
  ↓  SAM3 segmentation — shapes, arrows, icons, background
  ↓  VLM text labeling — per-shape OCR via local vision model
  ↓  Color, stroke, and fill extraction
  ↓  Arrow direction and endpoint detection
  ↓  DrawIO XML assembly
Output (.drawio file)
```

## Demo

| Input | Output |
|-------|--------|
| <img src="static/demo/original_1.jpg" width="340"/> | <img src="static/demo/recon_1.png" width="340"/> |
| <img src="static/demo/original_2.png" width="340"/> | <img src="static/demo/recon_2.png" width="340"/> |
| <img src="static/demo/original_3.jpg" width="340"/> | <img src="static/demo/recon_3.png" width="340"/> |

> Every shape, label, and arrow in the output is an independently editable element in DrawIO.

## Features

- **SAM3 segmentation** — fine-tuned on diagram elements; handles shapes, arrows, icons, and background regions
- **Local VLM text labeling** — crops each detected shape and sends it to a local vision model for OCR; no cloud API needed
- **Runs fully offline** — segmentation and text extraction both run on your own hardware
- **Apple Silicon support** — MPS fallback for ops not yet implemented on Metal
- **Auto-resize** — images larger than 1500 px on either axis are downscaled before segmentation
- **Segment cap** — keeps the top 80 highest-confidence segments to avoid slowdowns on dense diagrams
- **Live progress** — web UI shows each pipeline stage in real time

## Setup

### 1. Clone and install

```bash
git clone https://github.com/Kaleemullahqasim/openbanana.git
cd openbanana
pip install -r requirements.txt
```

> PyTorch must be installed separately — see [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).
> For torch 2.6.0 on macOS: `pip install torchvision==0.21.0`

### 2. Get model weights

| File | Where to get it | Put it at |
|------|-----------------|-----------|
| SAM3 checkpoint | [ModelScope — facebook/sam3](https://modelscope.cn/models/facebook/sam3) | `models/sam3/sam3.pt` |
| BPE vocab | bundled with the SAM3 download | `models/bpe_simple_vocab_16e6.txt.gz` |

### 3. Configure

```bash
cp config/config.yaml.example config/config.yaml
```

Open `config/config.yaml` and set:
- `sam3.checkpoint_path` — path to `sam3.pt`
- `sam3.bpe_path` — path to `bpe_simple_vocab_16e6.txt.gz`
- `multimodal.local_model` — your LM Studio model name (default: `allenai/olmocr-2-7b`)

### 4. (Optional) Local VLM for text labels

Text extraction uses a local vision model served by [LM Studio](https://lmstudio.ai/):

1. Install LM Studio and download `allenai/olmocr-2-7b`
2. Go to **Developer → Start Server**

If LM Studio is not running, the pipeline still completes — shapes will just have empty labels.

## Usage

### Web UI

```bash
# Apple Silicon (MPS)
PYTORCH_ENABLE_MPS_FALLBACK=1 python server_pa.py

# CUDA / CPU
python server_pa.py
```

Open [http://localhost:8000](http://localhost:8000), upload your image or PDF, and download the result.

### Command line

```bash
# Single image
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py -i input/diagram.png

# All images in input/
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py

# Skip text extraction
python main.py -i input/diagram.png --no-text

# Only specific element types
python main.py -i input/diagram.png --groups shape arrow

# With iterative refinement pass
python main.py -i input/diagram.png --refine
```

Results are saved to `output/<filename>/`.

## Architecture

The pipeline runs as a series of processors that share a `ProcessingContext` object:

```
main.py  →  ProcessingContext
             ├── TextRestorer          OCR pass on full image (Azure or VLM fallback)
             ├── Sam3InfoExtractor     SAM3 segmentation → ElementInfo list
             ├── IconPictureProcessor  Icons/images → base64 embedded in XML
             ├── BasicShapeProcessor   Color & stroke → DrawIO style strings
             ├── ArrowProcessor        Arrow direction & endpoint detection
             └── XMLMerger             Combine fragments by layer → .drawio output
```

Layer order (bottom → top): `BACKGROUND → BASIC_SHAPE → IMAGE → ARROW → TEXT`

Key files:

| File | Purpose |
|------|---------|
| `modules/data_types.py` | `ElementInfo`, `ProcessingContext`, `ElementType`, `LayerLevel` |
| `modules/sam3_info_extractor.py` | SAM3 client call, returns list of `ElementInfo` |
| `modules/vlm_labeler.py` | Per-shape VLM OCR |
| `modules/xml_merger.py` | Final DrawIO assembly |
| `sam3_service/server.py` | Standalone SAM3 inference service (FastAPI) |
| `config/config.yaml` | Runtime config |

## Roadmap

| Feature | Status |
|---------|--------|
| SAM3 segmentation + DrawIO export | ✅ Done |
| Local VLM text labeling | ✅ Done |
| Auto-resize for large images | ✅ Done |
| Live pipeline progress in web UI | ✅ Done |
| Segment confidence cap | ✅ Done |
| Smart arrow-to-shape connection | 🔄 In progress |
| PPTX export | 📍 Planned |
| Batch PDF processing | 📍 Planned |

## Contributing

Pull requests are welcome. For larger changes, open an issue first to discuss the approach.

```bash
git checkout -b feature/your-feature
# make changes
git commit -m 'feat: describe what you did'
git push origin feature/your-feature
# open a pull request
```

## Acknowledgements

OpenBanana builds on [SAM3](https://modelscope.cn/models/facebook/sam3) (Segment Anything Model 3) by Meta. The segmentation model used here was fine-tuned on diagram elements by the original Edit Banana research team at BIT.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
