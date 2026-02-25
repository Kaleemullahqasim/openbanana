<p align="center">
  <img src="/static/logo.svg" width="120" alt="OpenBanana Logo"/>
</p>

<h1 align="center">🍌 OpenBanana</h1>
<h3 align="center">Make the Uneditable, Editable</h3>

<p align="center">
  Convert static diagram images and PDFs into fully editable DrawIO XML — powered by a fine-tuned SAM3 segmentation model and local vision LLMs.
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-2F80ED?style=flat-square&logo=apache&logoColor=white" alt="License"/></a>
  <a href="https://developer.apple.com/metal/"><img src="https://img.shields.io/badge/Apple%20Silicon-MPS%20Supported-black?style=flat-square&logo=apple&logoColor=white" alt="MPS"/></a>
  <a href="https://developer.nvidia.com/cuda-downloads"><img src="https://img.shields.io/badge/GPU-CUDA%20Recommended-76B900?style=flat-square&logo=nvidia" alt="CUDA"/></a>
</p>

---

## What It Does

OpenBanana takes a PNG, JPG, or PDF of a diagram (flowchart, architecture diagram, UML, network diagram, etc.) and outputs a `.drawio` XML file where every shape, arrow, and label is an individually selectable, editable element.

**Pipeline:**
```
Input image / PDF
  → Preprocess (auto-resize if >1500px)
  → SAM3 segmentation  (shapes, arrows, icons, background)
  → VLM text labeling  (LM Studio / local vision model)
  → Color & stroke extraction
  → Arrow direction detection
  → DrawIO XML assembly
```

## Effect Demonstration

### Input → Output Comparison

| Scenario | Original (Static) | Reconstructed (Editable) |
|----------|-------------------|--------------------------|
| Basic Flowchart | <img src="/static/demo/original_1.jpg" width="360"/> | <img src="/static/demo/recon_1.png" width="360"/> |
| Multi-level Architecture | <img src="/static/demo/original_2.png" width="360"/> | <img src="/static/demo/recon_2.png" width="360"/> |
| Technical Schematic | <img src="/static/demo/original_3.jpg" width="360"/> | <img src="/static/demo/recon_3.png" width="360"/> |
| Formula Diagram | <img src="/static/demo/original_4.jpg" width="360"/> | <img src="/static/demo/recon_4.png" width="360"/> |

> Every output element is independently draggable, resizable, and re-styleable in DrawIO.

## Key Features

- **Fine-tuned SAM3** — custom mask decoder trained specifically on diagram elements (shapes, arrows, icons, background containers)
- **Local VLM text labeling** — crops each detected shape and queries a local vision model (LM Studio + `allenai/olmocr-2-7b`) to extract the text label; no cloud API required
- **Auto-resize** — images wider/taller than 1500 px are automatically downscaled before segmentation to prevent memory issues
- **Segment cap** — at most 80 highest-confidence elements are kept, preventing slowdowns on dense diagrams
- **Live progress** — the web UI shows real pipeline stages ("Segmenting…", "Extracting text labels…", "Building DrawIO XML…")
- **MPS support** — runs on Apple Silicon via `PYTORCH_ENABLE_MPS_FALLBACK`

## Installation

### 1. Clone & install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/open-banana.git
cd open-banana

pip install -r requirements.txt
# PyTorch must be installed separately — see https://pytorch.org/get-started/locally/
# For torch 2.6.0 on macOS: pip install torchvision==0.21.0
```

### 2. Download model weights

| Model | Source | Target path |
|-------|--------|-------------|
| SAM3 checkpoint | [ModelScope — facebook/sam3](https://modelscope.cn/models/facebook/sam3) | `models/sam3/sam3.pt` |
| BPE vocab | bundled with SAM3 | `models/bpe_simple_vocab_16e6.txt.gz` |

### 3. Configure

```bash
cp config/config.yaml.example config/config.yaml
```

Edit `config/config.yaml`:
- `sam3.checkpoint_path` — path to `sam3.pt`
- `sam3.bpe_path` — path to `bpe_simple_vocab_16e6.txt.gz`
- `multimodal.local_model` — LM Studio model name (default: `allenai/olmocr-2-7b`)

### 4. (Optional) Set up VLM text labeling

Text inside shapes is extracted using a local vision model via LM Studio:

1. Download and open [LM Studio](https://lmstudio.ai/)
2. Download `allenai/olmocr-2-7b` (OCR-specialized, fast)
3. Load it and start the server: **Developer tab → Start Server**

If LM Studio is not running, the pipeline still works — shapes will have empty labels.

## Usage

### Web interface (recommended)

```bash
# Apple Silicon
PYTORCH_ENABLE_MPS_FALLBACK=1 python server_pa.py

# CUDA
python server_pa.py
```

Open `http://localhost:8000`, upload an image or PDF, and download the `.drawio` result.

### CLI

```bash
# Single image
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py -i input/diagram.png

# Batch (all images in input/)
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py

# With iterative refinement
python main.py -i input/diagram.png --refine

# Skip OCR
python main.py -i input/diagram.png --no-text

# Specific element groups only
python main.py -i input/diagram.png --groups image arrow shape
```

Output is saved to `output/<image_stem>/`.

## Architecture

```
modules/
├── sam3_info_extractor.py   # SAM3 segmentation → ElementInfo list
├── basic_shape_processor.py # Color/stroke extraction → DrawIO style strings
├── arrow_processor.py       # Arrow direction + endpoint detection
├── icon_picture_processor.py# Icons encoded as base64
├── xml_merger.py            # Assemble fragments by LayerLevel → .drawio file
├── vlm_labeler.py           # VLM per-shape OCR (LM Studio)
├── data_types.py            # Shared types: ElementInfo, ProcessingContext, LayerLevel
└── text/
    └── restorer.py          # Azure OCR + VLM fallback (optional)

sam3/                        # SAM3 model library
sam3_service/                # Optional standalone SAM3 inference service
prompts/                     # Text prompts per element group
config/config.yaml           # Runtime config
```

**Layer ordering** (bottom → top): `BACKGROUND → BASIC_SHAPE → IMAGE → ARROW → TEXT`

## Development Roadmap

| Feature | Status |
|---------|--------|
| Core segmentation + DrawIO export | ✅ Done |
| Local VLM text labeling (LM Studio) | ✅ Done |
| Auto-resize large images | ✅ Done |
| Live progress in web UI | ✅ Done |
| Intelligent arrow-to-shape connection | 🔄 In progress |
| DrawIO template adaptation | 📍 Planned |
| Batch PDF export | 📍 Planned |

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'feat: add my feature'`
4. Push and open a Pull Request

Bug reports and feature requests: [Issues](../../issues)

## Contributors

| Name | Affiliation |
|------|-------------|
| Chai Chengliang | BIT |
| Zhang Chi | BIT |
| Deng Qiyan | |
| Rao Sijing | |
| Yi Xiangjian | |
| Li Jianhui | |
| Shen Chaoyuan | |
| Zhang Junkai | |
| Han Junyi | |
| You Zirui | |
| Xu Haochen | |
| An Minghao | |
| Yu Mingjie | |
| Yu Xinjiang | |
| Chen Zhuofan | |
| Li Xiangkun | |

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

---

⭐ If OpenBanana is useful to you, a star helps others find it!
