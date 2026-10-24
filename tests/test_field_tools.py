import io, subprocess, sys
from pathlib import Path
import numpy as np, pytest
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from test_core import leafy
from cropdx.import_field_dataset import build_index, map_folder
from cropdx.labels import PLANTVILLAGE_38
from cropdx.perturb import Perturbation, apply_background
from cropdx.preprocess import load_rgb_from_bytes, load_rgb_from_path, preprocess_pil


def _phone_jpeg(tmp_path, orientation=6):
    ex = Image.Exif(); ex[0x0112] = orientation
    p = tmp_path / "phone.jpg"; leafy(5).resize((320, 240)).save(p, "JPEG", quality=95, exif=ex.tobytes()); return p


def test_api_and_training_loaders_are_identical_incl_exif(tmp_path):
    p = _phone_jpeg(tmp_path)
    a = preprocess_pil(load_rgb_from_bytes(p.read_bytes())); b = preprocess_pil(load_rgb_from_path(p))
    assert np.array_equal(a, b)
    assert load_rgb_from_path(p).size == (240, 320), "EXIF rotation must be applied"


def test_non_square_is_cropped_not_squashed():
    wide = Image.new("RGB", (400, 200), (0, 0, 0)); wide.paste((255, 255, 255), (100, 0, 300, 200))  # white centre square
    x = preprocess_pil(wide)
    assert x.shape == (3, 224, 224) and np.allclose(x[0], x[0, 0, 0]), "centre square should fill the frame uniformly"


def test_matches_torchvision_pipeline():
    torch = pytest.importorskip("torch"); T = pytest.importorskip("torchvision.transforms")
    from cropdx.preprocess import MEAN, STD
    img = leafy(9)
    ref = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])(img).numpy()
    assert np.abs(ref - preprocess_pil(img)).max() < 1e-4        # no BGR / mean-std / resize mismatch


def test_real_background_and_shrink(tmp_path):
    bgdir = tmp_path / "bg"; bgdir.mkdir()
    Image.fromarray((np.random.default_rng(0).random((300, 400, 3)) * 255).astype(np.uint8)).save(bgdir / "soil.jpg")
    from cropdx.perturb import list_images
    paths = list_images(bgdir); assert len(paths) == 1
    rng = np.random.default_rng(1); src = leafy(2)
    changed = [apply_background(src, rng, None, paths, 1.0, 0.5) for _ in range(5)]
    assert all(c.size == src.size for c in changed)
    assert any(np.abs(np.asarray(c, np.int16) - np.asarray(src, np.int16)).mean() > 5 for c in changed)


def test_leaf_removed_leaves_only_background():
    src = leafy(3); out = Perturbation("leaf_removed")(src)
    assert np.asarray(out, float).std() < 0.5 * np.asarray(src, float).std()


def test_field_dataset_name_mapping():
    idx = build_index(PLANTVILLAGE_38)
    expect = {"Tomato Early blight leaf": "Tomato___Early_blight", "Tomato leaf late blight": "Tomato___Late_blight",
              "Tomato leaf": "Tomato___healthy", "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
              "Apple Scab Leaf": "Apple___Apple_scab", "Apple rust leaf": "Apple___Cedar_apple_rust",
              "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight", "Tomato mold leaf": "Tomato___Leaf_Mold",
              "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Potato leaf early blight": "Potato___Early_blight"}
    for folder, cid in expect.items():
        m = map_folder(folder, idx); assert m and m[0] == cid, (folder, m)
    assert map_folder("Banana leaf", idx) is None
