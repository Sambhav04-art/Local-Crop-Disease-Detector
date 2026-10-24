import io, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cropdx.dedupe import fingerprint, near_pairs
from cropdx.labels import PLANTVILLAGE_38, parse_class_name
from cropdx.perturb import PERTURBATIONS, Perturbation
from cropdx.preprocess import preprocess_pil


def test_label_parsing():
    assert parse_class_name("Tomato___Early_blight")["condition"] == "Early blight"
    assert parse_class_name("Pepper,_bell___healthy")["crop"] == "Pepper, bell"
    assert parse_class_name("Corn_(maize)___Common_rust_")["condition"] == "Common rust"
    assert parse_class_name("Tomato_healthy")["healthy"] and parse_class_name("Tomato__Target_Spot")["crop"] == "Tomato"
    assert len(set(PLANTVILLAGE_38)) == 38


def leafy(seed):
    r = np.random.default_rng(seed); a = np.full((256, 256, 3), 128, np.uint8)
    yy, xx = np.mgrid[:256, :256]
    for _ in range(6):
        cy, cx, rad = r.integers(60, 200, 2).tolist() + [int(r.integers(20, 50))]
        a[(yy - cy) ** 2 + (xx - cx) ** 2 < rad ** 2] = r.integers(20, 200, 3)
    return Image.fromarray(a)


def test_dedupe_finds_flips_rotations_and_recompression_but_not_others():
    base = [leafy(s) for s in range(30)]
    imgs = list(base)
    imgs.append(base[3].transpose(Image.FLIP_LEFT_RIGHT))      # idx 30
    imgs.append(base[5].rotate(90))                             # idx 31
    buf = io.BytesIO(); base[7].save(buf, "JPEG", quality=40); imgs.append(Image.open(io.BytesIO(buf.getvalue())))  # 32
    fps = [fingerprint(i.convert("L")) for i in imgs]
    thumbs, hashes = np.stack([f[0] for f in fps]), np.stack([f[1] for f in fps])
    a, b, mad = near_pairs(hashes, thumbs, 6)
    close = {frozenset((int(i), int(j))) for i, j, m in zip(a, b, mad) if m <= 5}
    assert {frozenset(p) for p in [(3, 30), (5, 31), (7, 32)]} <= close
    assert all(30 <= max(p) for p in close), "unrelated images were matched"


def test_preprocess_and_perturbations():
    x = preprocess_pil(leafy(1)); assert x.shape == (3, 224, 224) and x.dtype == np.float32
    for p in PERTURBATIONS:
        out = Perturbation(p)(leafy(2)); assert out.size == (256, 256), p
