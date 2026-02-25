"""
VLM per-shape text labeling using LM Studio.

Crops each detected diagram element and queries a local vision model
via LM Studio's OpenAI-compatible API with structured output (JSON schema)
to reliably extract the text/formula written inside that shape.

Recommended model: allenai/olmocr-2-7b  (OCR-specialized, fast, accurate)
Alternatives: qwen2.5vl-7b, minicpm-v, any vision model loaded in LM Studio

Setup:
    1. Open LM Studio
    2. Download + load: allenai/olmocr-2-7b
    3. Start server: Developer tab → Start Server  (or: lms server start)

Config (config/config.yaml → multimodal section):
    local_base_url: "http://localhost:1234/v1"
    local_api_key:  "lm-studio"
    local_model:    "allenai/olmocr-2-7b"
"""

import base64
import io
import json
from typing import Optional, List

from .data_types import ElementInfo


# ------------------------------------------------------------------
# Structured output schema — forces model to return {"text": "..."}
# ------------------------------------------------------------------
_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "diagram_text",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"}
            },
            "required": ["text"],
        },
    },
}

_SYSTEM = (
    "You are a specialized OCR engine for technical diagram elements. "
    "The images you receive are cropped regions from software architecture diagrams, "
    "flowcharts, UML diagrams, network diagrams, and similar technical drawings.\n\n"
    "Rules:\n"
    "- Transcribe text exactly as written — never correct spelling, casing, or abbreviations\n"
    "- For mathematical expressions or formulas, wrap in \\( ... \\) notation: e.g. \\(E = mc^2\\)\n"
    "- For multi-line text, join lines with \\n\n"
    "- If no text is visible, or you cannot read it clearly, return an empty string\n"
    "- Never invent or guess text that is not clearly visible in the image\n"
    "- Respond only with a JSON object containing exactly one field: \"text\""
)

_PROMPT = (
    "Examine this diagram element and extract all visible text exactly as written.\n\n"
    "Common content you might see:\n"
    "- Node/component labels: \"User Service\", \"Database\", \"API Gateway\"\n"
    "- Process labels: \"Validate Input\", \"Send Email\", \"Parse Response\"\n"
    "- Variable or field names: \"userId\", \"req.body\", \"error_code\"\n"
    "- Numeric values: port numbers, counts, percentages, IDs\n"
    "- Math expressions: output as \\(formula\\) notation\n"
    "- Status labels: \"Active\", \"Pending\", \"Error\", \"True\"\n\n"
    "Return exactly: {\"text\": \"<verbatim text>\"}\n"
    "If the element contains no readable text, return: {\"text\": \"\"}"
)

# ------------------------------------------------------------------
# Filtering constants
# ------------------------------------------------------------------
_SKIP_TYPES = {'arrow', 'line', 'connector'}

# Elements larger than this fraction of the canvas are background containers
# — they shouldn't be labeled with all the inner content they enclose.
_MAX_AREA_RATIO = 0.30

# Downscale large crops before sending to the model
_MAX_CROP_DIM = 512

# Responses that mean "no text found"
_EMPTY = {
    '', 'none', 'n/a', 'no text', 'empty', 'nothing', 'null',
    '(empty)', 'no content', 'no visible text', 'no text found',
}


class VLMLabeler:
    """
    For each non-arrow element, crop its bounding box from the input image
    and ask a local LM Studio vision model to OCR the text inside it.
    Stores the result in elem.text_label.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: the ``multimodal`` section from config/config.yaml
        """
        self.base_url = config.get('local_base_url', 'http://localhost:1234/v1')
        self.api_key  = config.get('local_api_key',  'lm-studio')
        self.model    = config.get('local_model',    'allenai/olmocr-2-7b')
        self._client   = None
        self._available = None   # cached after first call to is_available()

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            import httpx
            # Bypass system proxy (Clash/VPN/etc.) for localhost connections
            http_client = httpx.Client(proxy=None)
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                http_client=http_client,
            )
        return self._client

    def is_available(self) -> bool:
        """Return True if LM Studio is running and a matching model is loaded."""
        if self._available is not None:
            return self._available

        if not self.model or not self.model.strip():
            print("   VLM: no local_model configured, skipping")
            self._available = False
            return False

        try:
            client = self._get_client()
            models = client.models.list()
            ids = [m.id for m in models.data]

            # Loose match: "allenai/olmocr-2-7b" should match versioned/quantized IDs
            def _matches(m_id: str) -> bool:
                return (
                    self.model in m_id
                    or m_id.startswith(self.model)
                    or self.model.split('/')[-1] in m_id   # "olmocr-2-7b" in id
                )

            matched = [m_id for m_id in ids if _matches(m_id)]
            if matched:
                self.model = matched[0]   # use exact LM Studio model ID
                self._available = True
                print(f"   VLM: '{self.model}' ready in LM Studio ✓")
            else:
                available_str = ', '.join(ids[:5]) or '(none loaded)'
                print(f"   VLM: '{self.model}' not found in LM Studio.")
                print(f"        Loaded models: {available_str}")
                print(f"        → Load the model in LM Studio and start the server.")
                self._available = False

        except Exception as exc:
            print(f"   VLM: LM Studio not reachable ({exc})")
            print(f"        → Open LM Studio → Developer tab → Start Server")
            self._available = False

        return self._available

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def label_elements(
        self,
        elements: List[ElementInfo],
        image_path: str,
        canvas_width: int = 0,
        canvas_height: int = 0,
    ) -> int:
        """
        OCR text labels for all shape elements; store result in elem.text_label.

        Args:
            elements:              ElementInfo list from SAM3
            image_path:            original input image path
            canvas_width/height:   used to skip very large background containers

        Returns:
            Number of elements that received a non-empty text_label
        """
        if not self.is_available():
            return 0

        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        canvas_area = canvas_width * canvas_height

        labeled       = 0
        skipped_type  = 0
        skipped_size  = 0
        errors        = 0

        for elem in elements:
            etype = elem.element_type.lower()

            if etype in _SKIP_TYPES:
                skipped_type += 1
                continue

            if canvas_area > 0 and elem.bbox.area / canvas_area > _MAX_AREA_RATIO:
                skipped_size += 1
                continue

            try:
                label = self._label_element(elem, img)
                if label:
                    elem.text_label = label
                    labeled += 1
            except Exception:
                errors += 1

        suffix = f", {errors} errors)" if errors else ")"
        print(
            f"   VLM: labeled {labeled} elements"
            f" (skipped {skipped_type} connectors, {skipped_size} large containers{suffix}"
        )
        return labeled

    # ------------------------------------------------------------------
    # Per-element inference
    # ------------------------------------------------------------------

    def _label_element(self, elem: ElementInfo, img) -> Optional[str]:
        """Crop elem bbox → send to VLM → return text label or None."""
        pad = 4
        x1 = max(0, elem.bbox.x1 - pad)
        y1 = max(0, elem.bbox.y1 - pad)
        x2 = min(img.width,  elem.bbox.x2 + pad)
        y2 = min(img.height, elem.bbox.y2 + pad)

        crop = img.crop((x1, y1, x2, y2))
        if crop.width < 15 or crop.height < 15:
            return None

        # Downscale oversized crops for inference speed
        if crop.width > _MAX_CROP_DIM or crop.height > _MAX_CROP_DIM:
            crop = crop.copy()
            crop.thumbnail((_MAX_CROP_DIM, _MAX_CROP_DIM))

        # Encode as base64 PNG
        buf = io.BytesIO()
        crop.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        client = self._get_client()
        messages = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": _PROMPT},
                ],
            },
        ]

        # -- Try structured output first (LM Studio ≥ 0.3.x supports JSON schema) --
        result = None
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=_RESPONSE_FORMAT,
                max_tokens=150,
                timeout=30,
            )
            raw = response.choices[0].message.content.strip()
            result = json.loads(raw).get('text', '').strip()
        except Exception:
            pass   # fall through to plain-text fallback

        # -- Fallback: plain completion, parse JSON manually if present --
        if result is None:
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=150,
                    timeout=30,
                )
                raw = response.choices[0].message.content.strip()
                # If the model still returned JSON, parse it
                if raw.startswith('{'):
                    try:
                        result = json.loads(raw).get('text', raw).strip()
                    except Exception:
                        result = raw
                else:
                    result = raw
            except Exception:
                return None

        if not result or result.lower() in _EMPTY:
            return None

        return result
