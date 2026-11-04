# -*- coding: utf-8 -*-
"""
Best-of-N augmentations.

Hughes et al. (2024) show ASR scales as a power law in the number of stochastic
augmentations N. The mechanism: safety behavior is brittle to small, meaning-
preserving perturbations, so resampling many variants of the SAME payload finds
one that slips through. These are cheap, semantics-preserving perturbations of
the text and image channels, used by the `bon` strategy together with a
target temperature > 0 to generate N independent draws per config.
"""

from __future__ import annotations
import os
import random
from typing import List, Optional

# Confusable homoglyphs (meaning-preserving to a human reader).
_HOMOGLYPHS = {
    "a": "а", "e": "е", "o": "о", "c": "с",
    "p": "р", "x": "х", "y": "у", "i": "і",
}
_ZWSP = "​"  # zero-width space


def augment_text(text: str, rng: random.Random, intensity: float = 0.06) -> str:
    """Apply random meaning-preserving text perturbations.

    Mixes: homoglyph substitution, zero-width-space insertion, random casing,
    and light whitespace jitter. `intensity` ~ fraction of characters touched.
    """
    if not text:
        return text
    chars = list(text)
    n = len(chars)
    k = max(1, int(n * intensity))
    idxs = rng.sample(range(n), min(k, n))
    for i in idxs:
        ch = chars[i]
        r = rng.random()
        low = ch.lower()
        if r < 0.4 and low in _HOMOGLYPHS:
            chars[i] = _HOMOGLYPHS[low] if ch.islower() else _HOMOGLYPHS[low].upper()
        elif r < 0.7 and ch.isalpha():
            chars[i] = ch.upper() if ch.islower() else ch.lower()
        elif r < 0.9 and ch == " ":
            chars[i] = "  "  # double space
        else:
            chars[i] = ch + _ZWSP
    return "".join(chars)


def augment_image(image_path: str, rng: random.Random, out_dir: str,
                  idx: int = 0) -> Optional[str]:
    """Apply random meaning-preserving image perturbations and save a variant.

    Mixes: small rotation, crop/pad, JPEG recompression at random quality, and
    brightness/contrast jitter. Returns the path of the perturbed image, or the
    original path if PIL is unavailable / the image is missing.
    """
    if not image_path or not os.path.isfile(image_path):
        return image_path
    try:
        from PIL import Image, ImageEnhance
    except Exception:
        return image_path
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        # Small rotation.
        if rng.random() < 0.6:
            img = img.rotate(rng.uniform(-5, 5), expand=False, fillcolor=(255, 255, 255))

        # Random crop then resize back (scale jitter).
        if rng.random() < 0.6:
            cx = int(w * rng.uniform(0.0, 0.06))
            cy = int(h * rng.uniform(0.0, 0.06))
            img = img.crop((cx, cy, w - cx, h - cy)).resize((w, h))

        # Brightness / contrast jitter.
        if rng.random() < 0.6:
            img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.9, 1.1))
        if rng.random() < 0.6:
            img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.9, 1.1))

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"bon_aug_{idx:03d}.jpg")
        quality = rng.randint(55, 92)  # JPEG recompression
        img.save(out_path, "JPEG", quality=quality)
        return out_path
    except Exception:
        return image_path
