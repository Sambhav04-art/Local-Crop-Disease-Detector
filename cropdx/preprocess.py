"""Single source of truth for inference preprocessing (used by training eval AND the API)."""
import io
import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
Image.MAX_IMAGE_PIXELS = 60_000_000  # decompression-bomb guard
DRAFT = (640, 640)  # big phone JPEGs are decoded at reduced size (fast); identical in training and API


def _finish(img: Image.Image) -> Image.Image:
    if img.format == "JPEG":
        img.draft("RGB", DRAFT)         # big phone JPEGs are decoded at reduced size (fast)
    img.load()
    img = ImageOps.exif_transpose(img)  # phone photos carry rotation in EXIF
    return img.convert("RGB")


def load_rgb_from_bytes(data: bytes) -> Image.Image:
    return _finish(Image.open(io.BytesIO(data)))


def load_rgb_from_path(path) -> Image.Image:
    """Used by training/evaluation. Identical to the API's loader (parity is unit-tested)."""
    with Image.open(path) as im:
        return _finish(im)


def preprocess_pil(img: Image.Image, size=IMG_SIZE, mean=MEAN, std=STD) -> np.ndarray:
    """RGB PIL -> float32 CHW array, resized (bilinear) to size x size and normalised."""
    img = img.convert("RGB")
    w, h = img.size
    if w != h:  # centre-crop to a square instead of squashing a 4:3 photo (keep the leaf in the middle!)
        side = min(w, h); left, top = (w - side) // 2, (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(a.transpose(2, 0, 1), dtype=np.float32)


def quality_warnings(img: Image.Image) -> list:
    """Cheap heuristics (not a guarantee) to tell the user why a photo may be unreliable."""
    w, h = img.size
    out = []
    if min(w, h) < 150:
        out.append("LOW_RESOLUTION")
    g = np.asarray(img.convert("L").resize((256, 256), Image.BILINEAR), dtype=np.float32)
    m = float(g.mean())
    if m < 45:
        out.append("IMAGE_TOO_DARK")
    elif m > 215:
        out.append("IMAGE_TOO_BRIGHT")
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    if float(lap.var()) < 40:
        out.append("IMAGE_BLURRY")
    return out
