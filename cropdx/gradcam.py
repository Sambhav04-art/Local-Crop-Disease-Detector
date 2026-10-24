"""Where is the model looking?   python -m cropdx.gradcam --images "C:\\photos" [--model auto] [--n 12]
Red/yellow = regions that drive the prediction. If it lights up soil, hands or background instead of the leaf/lesions => background bias."""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import load_config, path_of
from .evaluate import pick_best
from .models import build_model
from .preprocess import MEAN, STD, load_rgb_from_path, preprocess_pil


def target_layer(model, name):
    return model.layer4[-1] if name == "resnet50" else model.features[-1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--images", required=True); ap.add_argument("--model", default="auto")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--out")
    a = ap.parse_args(); cfg = load_config()
    name = pick_best(path_of(cfg, "runs_dir")) if a.model == "auto" else a.model
    ck = torch.load(path_of(cfg, "runs_dir") / name / "best.pt", map_location="cpu")
    classes = ck["classes"]; size = ck["image_size"]
    model = build_model(name, len(classes), pretrained=False); model.load_state_dict(ck["state_dict"]); model.eval()
    feats, grads = {}, {}
    layer = target_layer(model, name)
    def fwd_hook(m, i, o):          # tensor hook (module backward hooks break with in-place activations)
        feats["a"] = o
        o.register_hook(lambda g: grads.__setitem__("g", g))
    layer.register_forward_hook(fwd_hook)
    root = Path(a.images)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    files = files[:a.n]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cols = 4; rows = (len(files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4)); axes = np.atleast_1d(axes).ravel()
    for ax in axes: ax.axis("off")
    for ax, f in zip(axes, files):
        x = torch.from_numpy(preprocess_pil(load_rgb_from_path(f), size)).unsqueeze(0).clone().requires_grad_(True)
        model.zero_grad(); logits = model(x); k = int(logits.argmax(1)); conf = float(F.softmax(logits, 1)[0, k])
        logits[0, k].backward()
        w = grads["g"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * feats["a"].detach()).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)[0, 0].detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        img = (x[0].detach().numpy().transpose(1, 2, 0) * np.array(STD) + np.array(MEAN)).clip(0, 1)
        ax.imshow(img); ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.set_title(f"{classes[k].replace('___', ': ')[:30]}\nraw softmax {conf:.2f}", fontsize=7)
    out = Path(a.out) if a.out else path_of(cfg, "reports_dir") / "gradcam.png"
    fig.tight_layout(); fig.savefig(out, dpi=110); print("Saved", out)


if __name__ == "__main__":
    main()
