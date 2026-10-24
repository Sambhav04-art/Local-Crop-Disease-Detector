"""Synthetic 'real-world' degradations: used as training augmentation AND for stress-testing."""
import io
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

PALETTES = {  # base RGB of procedural backgrounds
    "soil": (110, 80, 55), "grass": (70, 115, 50), "sand": (170, 150, 110),
    "gray": (120, 120, 125), "foliage": (30, 60, 30), "sky": (150, 180, 210),
}
_PAL_LIST = list(PALETTES)


def leaf_mask(img: Image.Image):
    """Rough leaf/background separation for PlantVillage-style photos (uniform background).
    Returns an 'L' mask, or None when the background is not uniform (e.g. a real field photo)."""
    a = np.asarray(img.convert("RGB"), dtype=np.int16)
    ring = np.concatenate([a[:6].reshape(-1, 3), a[-6:].reshape(-1, 3),
                           a[:, :6].reshape(-1, 3), a[:, -6:].reshape(-1, 3)])
    if ring.std(axis=0).mean() > 25:
        return None
    bg = np.median(ring, axis=0)
    m = (np.abs(a - bg).sum(axis=2) > 45).astype(np.uint8) * 255
    mask = Image.fromarray(m).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    cov = float((np.asarray(mask) > 127).mean())
    if cov < 0.08 or cov > 0.97:
        return None
    return mask.filter(ImageFilter.GaussianBlur(1.5))


def random_background(size, rng, palette=None):
    w, h = size
    name = palette or _PAL_LIST[int(rng.integers(len(_PAL_LIST)))]
    base = np.array(PALETTES[name], np.float32) + rng.integers(-25, 25, 3)
    field = np.zeros((h, w), np.float32)
    for scale, amp in ((4, 1.0), (12, 0.6), (40, 0.3)):
        small = Image.fromarray((rng.random((scale, scale)) * 255).astype(np.uint8))
        field += (np.asarray(small.resize((w, h), Image.BICUBIC), np.float32) / 255 - 0.5) * amp
    arr = base[None, None, :] + field[..., None] * float(rng.uniform(40, 90))
    arr += rng.normal(0, 4, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


_BG_POOL = {}   # per-process cache: generating backgrounds is the costly part, so reuse + flip them
_BG_POOL_SIZE = 24


def _pooled_background(size, rng, palette):
    key = (size, palette)
    pool = _BG_POOL.setdefault(key, [])
    if len(pool) < _BG_POOL_SIZE:
        pool.append(random_background(size, rng, palette))
        return pool[-1]
    bg = pool[int(rng.integers(len(pool)))]
    if rng.random() < 0.5:
        bg = bg.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        bg = bg.transpose(Image.FLIP_TOP_BOTTOM)
    return bg


IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def list_images(folder):
    if not folder or not Path(folder).is_dir():
        return []
    return sorted(str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)


def _real_background(size, rng, paths):
    """Random square crop of a REAL photo (soil, grass, hands, tables...) used as background."""
    with Image.open(paths[int(rng.integers(len(paths)))]) as im:
        if im.format == "JPEG":
            im.draft("RGB", (768, 768))
        im = im.convert("RGB")
    w, h = im.size
    crop = max(16, int(min(w, h) * float(rng.uniform(0.35, 1.0))))
    x0, y0 = int(rng.integers(0, w - crop + 1)), int(rng.integers(0, h - crop + 1))
    out = im.crop((x0, y0, x0 + crop, y0 + crop)).resize(size, Image.BILINEAR)
    return out.transpose(Image.FLIP_LEFT_RIGHT) if rng.random() < 0.5 else out


def apply_background(img, rng, palette=None, bg_paths=None, real_prob=0.0, shrink=None):
    """Cut the leaf out of its plain lab background and paste it on a new background.
    shrink=0.55 means the leaf may be scaled to 55-100% and placed anywhere (leaf far from camera)."""
    mask = leaf_mask(img)
    if mask is None:
        return img
    img = img.convert("RGB")
    bg = None
    if bg_paths and rng.random() < real_prob:
        try:
            bg = _real_background(img.size, rng, bg_paths)
        except Exception:
            bg = None
    if bg is None:
        bg = _pooled_background(img.size, rng, palette)
    if shrink and shrink < 1.0 and rng.random() < 0.7:
        s = float(rng.uniform(shrink, 1.0)); w, h = img.size
        nw, nh = max(8, int(w * s)), max(8, int(h * s))
        ox, oy = int(rng.integers(0, w - nw + 1)), int(rng.integers(0, h - nh + 1))
        canvas = bg.copy()
        canvas.paste(img.resize((nw, nh), Image.BILINEAR), (ox, oy), mask.resize((nw, nh), Image.BILINEAR))
        return canvas
    return Image.composite(img, bg, mask)


def jpeg_compress(img, quality):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=int(quality))
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out.convert("RGB")


def _rng_from(img):  # deterministic per image, safe with DataLoader workers
    return np.random.default_rng(int(np.asarray(img.resize((4, 4), Image.BOX)).sum()))


class RandomBackground:
    def __init__(self, p=0.3, bg_dir=None, real_prob=0.7, shrink=0.55):
        self.p, self.real_prob, self.shrink = p, real_prob, shrink
        self.paths = list_images(bg_dir)

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        rng = np.random.default_rng(random.getrandbits(32))
        return apply_background(img, rng, None, self.paths, self.real_prob, self.shrink)


class RandomJPEG:
    def __init__(self, p=0.3, qmin=15, qmax=60):
        self.p, self.qmin, self.qmax = p, qmin, qmax

    def __call__(self, img):
        return jpeg_compress(img, random.randint(self.qmin, self.qmax)) if random.random() < self.p else img


SHORTCUT_TEST = "leaf_removed"                       # diagnostic, NOT a normal stress test
REAL_BG_TESTS = ["bg_real_heldout", "small_leaf_real_bg"]   # need data/backgrounds_test/
PERTURBATIONS = ["dark", "overexposed", "low_contrast", "warm_cast", "cool_cast", "blur", "noise",
                 "low_res", "jpeg_q10", "close_up", "bg_soil", "bg_grass", "bg_gray", "field_combo"] + REAL_BG_TESTS


def _scale(img, f):
    a = np.asarray(img, np.float32) * np.asarray(f, np.float32)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


class Perturbation:
    """Picklable (Windows-safe) callable: PIL -> degraded PIL."""

    def __init__(self, name, bg_paths=None):
        assert name in PERTURBATIONS or name == SHORTCUT_TEST, name
        self.name, self.bg_paths = name, bg_paths or []

    def __call__(self, img):
        n, img = self.name, img.convert("RGB")
        w, h = img.size
        if n == "dark":
            return ImageEnhance.Brightness(img).enhance(0.35)
        if n == "overexposed":
            return ImageEnhance.Brightness(img).enhance(1.9)
        if n == "low_contrast":
            return ImageEnhance.Contrast(img).enhance(0.45)
        if n == "warm_cast":
            return _scale(img, (1.25, 1.0, 0.8))
        if n == "cool_cast":
            return _scale(img, (0.8, 1.0, 1.25))
        if n == "blur":
            return img.filter(ImageFilter.GaussianBlur(3))
        if n == "noise":
            a = np.asarray(img, np.float32) + _rng_from(img).normal(0, 20, (h, w, 3))
            return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        if n == "low_res":
            return img.resize((48, 48), Image.BILINEAR).resize((w, h), Image.BILINEAR)
        if n == "jpeg_q10":
            return jpeg_compress(img, 10)
        if n == "close_up":
            return img.crop((int(.225 * w), int(.225 * h), int(.775 * w), int(.775 * h))).resize((w, h), Image.BICUBIC)
        if n == SHORTCUT_TEST:   # paint the leaf with the background colour: only background cues remain
            mask = leaf_mask(img)
            if mask is None:
                return img
            a = np.asarray(img, np.int16)
            ring = np.concatenate([a[:6].reshape(-1, 3), a[-6:].reshape(-1, 3), a[:, :6].reshape(-1, 3), a[:, -6:].reshape(-1, 3)])
            fill = Image.new("RGB", img.size, tuple(int(v) for v in np.median(ring, axis=0)))
            return Image.composite(fill, img, mask)
        if n == "bg_real_heldout":
            return apply_background(img, _rng_from(img), None, self.bg_paths, 1.0, None)
        if n == "small_leaf_real_bg":
            return apply_background(img, _rng_from(img), None, self.bg_paths, 1.0, 0.45)
        if n.startswith("bg_"):
            return apply_background(img, _rng_from(img), n[3:])
        if n == "field_combo":  # background + dim light + soft focus + compression
            out = apply_background(img, _rng_from(img))
            out = ImageEnhance.Brightness(out).enhance(0.65).filter(ImageFilter.GaussianBlur(1.5))
            return jpeg_compress(out, 30)
        raise ValueError(n)
