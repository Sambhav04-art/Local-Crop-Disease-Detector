"""STEP 4: export ONNX + labels + metadata into artifacts/ (what the API loads).  python -m cropdx.export"""
import argparse, datetime, json
import numpy as np
import pandas as pd
import torch

from .config import load_config, path_of
from .constants import DISCLAIMER
from .data import EvalTransform
from .models import build_model
from .preprocess import MEAN, STD, preprocess_pil


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--version")
    a = ap.parse_args(); cfg = load_config(a.config)
    dep = json.loads((path_of(cfg, "reports_dir") / "deployment_config.json").read_text())
    classes = json.loads((path_of(cfg, "processed_dir") / "classes.json").read_text())
    out = path_of(cfg, "artifacts_dir"); out.mkdir(parents=True, exist_ok=True)
    size = cfg["image"]["size"]
    ck = torch.load(path_of(cfg, "runs_dir") / dep["model"] / "best.pt", map_location="cpu")
    model = build_model(dep["model"], len(classes), pretrained=False); model.load_state_dict(ck["state_dict"]); model.eval()
    dummy = torch.randn(1, 3, size, size)
    kw = dict(input_names=["input"], output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
              opset_version=cfg["export"]["opset"])
    try:
        torch.onnx.export(model, dummy, str(out / "model.onnx"), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, dummy, str(out / "model.onnx"), **kw)

    # parity check torch vs onnxruntime on real validation images
    import onnxruntime as ort
    from PIL import Image
    sess = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])
    df = pd.read_csv(path_of(cfg, "processed_dir") / "manifest.csv"); df = df[df.split == "val"].sample(16, random_state=0)
    x = np.stack([preprocess_pil(Image.open(p).convert("RGB"), size) for p in df.path])
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    got = sess.run(None, {"input": x})[0]
    diff = float(np.abs(ref - got).max()); print("max |torch - onnx| logit diff:", diff)
    assert diff < 1e-2 and (ref.argmax(1) == got.argmax(1)).all(), "ONNX export mismatch"

    (out / "labels.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    meta = {"modelVersion": a.version or cfg["export"]["model_version"], "architecture": dep["model"], "imageSize": size,
            "mean": list(MEAN), "std": list(STD), "temperature": dep["temperature"],
            "confidenceThreshold": dep["confidence_threshold"], "numClasses": len(classes),
            "testAccuracy": dep["test_accuracy"], "testMacroF1": dep["test_macro_f1"],
            "trainedDataset": "PlantVillage (lab-condition leaf images)",
            "exportedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(), "disclaimer": DISCLAIMER}
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print("Exported to", out)


if __name__ == "__main__":
    main()
