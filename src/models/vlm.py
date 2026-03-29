"""
Vision Language Model (VLM) wrapper.
Uses LLaVA via Ollama to generate engineering-context descriptions
of images extracted from CAE/FEA PDF documents.
"""

import base64
from typing import Optional

import httpx

from src.config import OLLAMA_BASE_URL, VLM_MODEL


_CAE_IMAGE_PROMPT = """You are an expert in Finite Element Analysis (FEA) and \
Computer-Aided Engineering (CAE). Analyse the provided engineering image and describe:

1. What type of image/plot this is (e.g., FEA stress contour plot, load-displacement \
curve, bolt geometry diagram, material microstructure, failure mode illustration, etc.)
2. Key values, labels, or annotations visible in the image
3. What engineering information or insight this image communicates
4. Any colour scales, axes, legends, or units present

Be precise and technical. Your description will be used to answer engineering queries \
from analysts who cannot see the original image.
Source document: {source_filename}
"""


def summarise_image(
    image_b64: str,
    source_filename: str = "unknown",
    prompt: Optional[str] = None,
) -> str:
    """
    Send a base64-encoded image to LLaVA via the Ollama API and return
    a structured engineering description.

    Args:
        image_b64: Base64-encoded PNG/JPEG bytes of the image.
        source_filename: Name of the PDF this image was extracted from.
        prompt: Optional custom prompt. Falls back to CAE-specific default.

    Returns:
        A string description of the image content.

    Raises:
        RuntimeError: If the Ollama API call fails.
    """
    prompt_text = prompt or _CAE_IMAGE_PROMPT.format(source_filename=source_filename)

    payload = {
        "model": VLM_MODEL,
        "prompt": prompt_text,
        "images": [image_b64],
        "stream": False,
    }

    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=600.0,  # VLM inference can be slow on CPU
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except httpx.HTTPError as exc:
        return f"[Image from {source_filename} - VLM unavailable: {exc}]"
