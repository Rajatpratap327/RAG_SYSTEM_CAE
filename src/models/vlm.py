"""
Vision Language Model (VLM) wrapper.

Uses Ollama VLM (MiniCPM-V / LLaVA) to generate engineering-context
descriptions of images extracted from CAE/FEA PDF documents.
"""

import base64
import io
from typing import Optional

import httpx
from PIL import Image

from src.config import OLLAMA_BASE_URL, VLM_MODEL


# ---------------------------------------------------------------------
# Persistent HTTP client (improves stability + performance)
# ---------------------------------------------------------------------
_client = httpx.Client(timeout=300.0)


# ---------------------------------------------------------------------
# Engineering Prompt
# ---------------------------------------------------------------------
_CAE_IMAGE_PROMPT = """You are an expert in Finite Element Analysis (FEA) and
Computer-Aided Engineering (CAE). Analyse the provided engineering image and describe:

1. What type of image/plot this is (e.g., FEA stress contour plot,
   load-displacement curve, bolt geometry diagram, material microstructure,
   failure mode illustration, etc.)
2. Key values, labels, or annotations visible in the image
3. What engineering information or insight this image communicates
4. Any colour scales, axes, legends, or units present

Be precise, objective, and technical.
Do not speculate beyond visible information.

Source document: {source_filename}
"""


# ---------------------------------------------------------------------
# Image Resize Utility (CRITICAL for VLM stability)
# ---------------------------------------------------------------------
def _resize_image_b64(image_b64: str, max_size: int = 1024) -> str:
    """
    Resize large images before sending to VLM.

    Large CAE PDF images often cause Ollama timeouts because
    VLMs expect ~1024px images, not multi-megapixel renders.
    """

    try:
        image_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(image_bytes))

        # Convert unsupported modes
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        # Maintain aspect ratio
        img.thumbnail((max_size, max_size))

        buffer = io.BytesIO()
        img.save(buffer, format="PNG", optimize=True)

        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    except Exception:
        # If resize fails, fall back safely
        return image_b64


# ---------------------------------------------------------------------
# Main VLM Function
# ---------------------------------------------------------------------
def summarise_image(
    image_b64: str,
    source_filename: str = "unknown",
    prompt: Optional[str] = None,
) -> str:
    """
    Send a base64 image to Ollama VLM and return an engineering description.

    Args:
        image_b64: Base64-encoded PNG/JPEG image
        source_filename: Origin PDF filename
        prompt: Optional override prompt

    Returns:
        Technical description string
    """

    # Resize image to prevent CPU timeout
    image_b64 = _resize_image_b64(image_b64)

    prompt_text = prompt or _CAE_IMAGE_PROMPT.format(
        source_filename=source_filename
    )

    payload = {
        "model": VLM_MODEL,
        "prompt": prompt_text,
        "images": [image_b64],
        "stream": False,
        "options": {
            # safer context for vision models
            "num_ctx": 1024,
            # deterministic engineering output
            "temperature": 0.1,
        },
    }

    try:
        response = _client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
        )

        response.raise_for_status()
        data = response.json()

        result = data.get("response", "").strip()

        # Guard against silent empty responses
        if not result:
            raise RuntimeError("Empty response from VLM")

        return result

    except Exception as exc:
        # Never crash ingestion pipeline
        return (
            f"[Image from {source_filename} - "
            f"VLM unavailable: {str(exc)}]"
        )