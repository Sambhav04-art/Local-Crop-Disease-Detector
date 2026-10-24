import io, json, subprocess, sys, shutil
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from test_core import leafy


def test_prepare_data_on_fake_dataset(tmp_path):
    for variant in ("color", "grayscale", "segmented"):
        for ci, cls in enumerate(["Tomato___Early_blight", "Tomato___healthy", "Potato___Late_blight"]):
            d = tmp_path / "raw" / "plantvillage dataset" / variant / cls; d.mkdir(parents=True)
            for k in range(60):
                im = leafy(ci * 1000 + k)
                im.convert("L" if variant == "grayscale" else "RGB").save(d / f"img{k}.JPG")
    c = tmp_path / "raw" / "plantvillage dataset" / "color" / "Tomato___healthy"
    leafy(1000).transpose(Image.FLIP_TOP_BOTTOM).save(c / "flipped_dup.JPG")   # duplicate of img0
    (c / "broken.JPG").write_bytes(b"not an image")
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    ft = tmp_path / "ft" / "Tomato___healthy"; ft.mkdir(parents=True)
    for k in range(3):
        leafy(9000 + k).save(ft / f"real{k}.jpg")
    (tmp_path / "ft" / "Banana___unknown").mkdir()
    cfg["paths"].update(data_root=str(tmp_path / "raw"), processed_dir=str(tmp_path / "processed_dir"),
                        reports_dir=str(tmp_path / "reports_dir"), field_train_dir=str(tmp_path / "ft"))
    p = tmp_path / "cfg.yaml"; p.write_text(yaml.safe_dump(cfg))
    subprocess.run([sys.executable, "-m", "cropdx.prepare_data", "--config", str(p)], check=True, cwd=ROOT)
    df = pd.read_csv(tmp_path / "processed_dir" / "manifest.csv")
    assert all("/color/" in x.replace("\\", "/") for x in df[df.source == "plantvillage"].path), "must use only the colour folder"
    rep = json.loads((tmp_path / "reports_dir" / "data_report.json").read_text())
    assert rep["removed_by_reason"]["corrupt_or_unreadable"] == 1
    assert rep["removed_by_reason"]["duplicate_of_kept_image"] >= 1
    pv = df[df.source == "plantvillage"]
    assert pv.groupby("group").split.nunique().max() == 1 and set(pv.split) == {"train", "val", "test"}
    fld = df[df.source == "field"]
    assert len(fld) == 3 and set(fld.split) == {"train"} and set(fld.class_name) == {"Tomato___healthy"}


def test_api_contract(monkeypatch):
    subprocess.run([sys.executable, str(ROOT / "tests" / "make_dummy_model.py")], check=True)
    monkeypatch.setenv("MODEL_DIR", str(ROOT / "artifacts_dummy")); monkeypatch.setenv("API_KEY", "secret")
    from fastapi.testclient import TestClient
    import importlib, api.main as m; importlib.reload(m)
    with TestClient(m.app) as c:
        assert c.get("/health").json() == {"status": "UP", "modelLoaded": True}
        h = {"X-API-Key": "secret"}
        assert c.get("/v1/classes").status_code == 401 and len(c.get("/v1/classes", headers=h).json()) == 38
        buf = io.BytesIO(); leafy(3).save(buf, "JPEG")
        r = c.post("/v1/predict?topK=3", headers=h, files={"file": ("leaf.jpg", buf.getvalue(), "image/jpeg")})
        j = r.json(); print(json.dumps(j, indent=2))
        assert r.status_code == 200 and len(j["candidates"]) == 3 and j["status"] in ("CONFIDENT", "LOW_CONFIDENCE")
        assert (j["prediction"] is None) == (j["status"] == "LOW_CONFIDENCE") and "requestId" in j and "disclaimer" in j
        bad = c.post("/v1/predict", headers=h, files={"file": ("x.jpg", b"garbage", "image/jpeg")})
        assert bad.status_code == 400 and bad.json()["error"]["code"] == "INVALID_IMAGE"
        assert c.post("/v1/predict", headers=h).status_code == 422
        assert c.post("/v1/predict", headers=h, files={"file": ("e.jpg", b"", "image/jpeg")}).json()["error"]["code"] == "EMPTY_FILE"
