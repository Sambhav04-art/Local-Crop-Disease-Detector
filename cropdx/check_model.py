"""Triage tool: run the DEPLOYED model (artifacts/model.onnx, exactly what the API serves) on a folder of photos.

  python -m cropdx.check_model --images "C:\\my_leaf_photos"
Put photos in sub-folders named like the class ids (e.g. Tomato___Early_blight/) to get accuracy; otherwise you get predictions only.
No PyTorch needed."""
import argparse, collections, json
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort

from .config import load_config, path_of
from .preprocess import load_rgb_from_path, preprocess_pil, quality_warnings

EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True); ap.add_argument("--artifacts"); ap.add_argument("--sheet", type=int, default=30)
    a = ap.parse_args()
    cfg = load_config(); art = Path(a.artifacts) if a.artifacts else path_of(cfg, "artifacts_dir"); rep = path_of(cfg, "reports_dir")
    rep.mkdir(parents=True, exist_ok=True)
    meta = json.loads((art / "model_meta.json").read_text()); classes = json.loads((art / "labels.json").read_text())
    ids = [c["classId"] for c in classes]; T, thr = meta["temperature"], meta["confidenceThreshold"]
    sess = ort.InferenceSession(str(art / "model.onnx"), providers=["CPUExecutionProvider"])
    print(f"Model: {meta['modelVersion']} ({meta['architecture']}) | threshold {thr} | lab test accuracy {meta['testAccuracy']:.3f}")
    if "DUMMY" in meta["modelVersion"].upper():
        print("!!! This is the DUMMY model (random output). Point MODEL_DIR/--artifacts at the real artifacts/ folder.")
    ms = path_of(cfg, "reports_dir") / "metrics_summary.json"
    if ms.exists() and json.loads(ms.read_text()).get("smoke_run"):
        print("!!! The last evaluation was a --smoke run: the model was only trained for 1 epoch on a tiny subset. Run the full pipeline.")

    root = Path(a.images)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXT)
    if not files:
        raise SystemExit(f"No images found in {root}")
    rows = []
    for f in files:
        img = load_rgb_from_path(f)
        z = sess.run(None, {"input": preprocess_pil(img, meta["imageSize"], meta["mean"], meta["std"])[None]})[0][0].astype(np.float64) / T
        z -= z.max(); p = np.exp(z); p /= p.sum(); o = np.argsort(-p)[:3]
        true = f.parent.name if f.parent.name in ids else None
        rows.append({"file": str(f), "true": true, "pred1": ids[o[0]], "conf1": round(float(p[o[0]]), 3),
                     "pred2": ids[o[1]], "conf2": round(float(p[o[1]]), 3), "pred3": ids[o[2]], "conf3": round(float(p[o[2]]), 3),
                     "status": "CONFIDENT" if p[o[0]] >= thr else "LOW_CONFIDENCE", "warnings": ",".join(quality_warnings(img)),
                     "top3_hit": (true in [ids[i] for i in o]) if true else None})
    df = pd.DataFrame(rows); df.to_csv(rep / "photo_check.csv", index=False)
    print(f"\n{len(df)} photos | CONFIDENT {int((df.status=='CONFIDENT').sum())} | LOW_CONFIDENCE {int((df.status=='LOW_CONFIDENCE').sum())} | mean confidence {df.conf1.mean():.2f}")
    top = df.pred1.value_counts(normalize=True).head(5)
    print("Most predicted classes:", {k: f"{v:.0%}" for k, v in top.items()})
    if len(df) >= 15 and top.iloc[0] > 0.4:
        print(f"!!! COLLAPSE: {top.iloc[0]:.0%} of all photos get the same label ({top.index[0]}). Typical of background/lighting shortcut learning or a domain gap.")
    w = df.warnings.str.split(",").explode().replace("", np.nan).dropna().value_counts()
    if len(w): print("Photo quality warnings:", w.to_dict())
    lab = df[df.true.notna()]
    if len(lab):
        ok = lab.true == lab.pred1
        print(f"\nLabelled photos: {len(lab)} | top-1 accuracy {ok.mean():.1%} | top-3 {lab.top3_hit.astype(bool).mean():.1%}")
        c = lab[lab.status == "CONFIDENT"]
        if len(c): print(f"Of CONFIDENT answers: {len(c)} ({len(c)/len(lab):.0%} of photos), accuracy {(c.true==c.pred1).mean():.1%}")
        print(f"Mean confidence when RIGHT {lab.conf1[ok].mean():.2f} | when WRONG {lab.conf1[~ok].mean() if (~ok).any() else float('nan'):.2f}  (close values = confidence is not trustworthy)")
        pairs = collections.Counter(zip(lab.true[~ok], lab.pred1[~ok])).most_common(6)
        print("Most common mistakes (true -> predicted):"); [print(f"  {n}x {t} -> {p}") for (t, p), n in pairs]
        print(f"\nCompare with lab test accuracy {meta['testAccuracy']:.1%}: a big gap = lab-to-field domain shift.")
    # contact sheet: mistakes first
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    order = df.assign(wrong=(df.true.notna() & (df.true != df.pred1)), low=df.status == "LOW_CONFIDENCE").sort_values(["wrong", "low"], ascending=False).head(a.sheet)
    cols = 6; rws = (len(order) + cols - 1) // cols
    fig, axes = plt.subplots(rws, cols, figsize=(cols * 2.6, rws * 2.9)); axes = np.atleast_1d(axes).ravel()
    for ax in axes: ax.axis("off")
    for ax, (_, r) in zip(axes, order.iterrows()):
        im = load_rgb_from_path(r.file); im.thumbnail((256, 256)); ax.imshow(im)
        col = "k" if r.true is None else ("green" if r.true == r.pred1 else "red")
        ax.set_title(f"{r.pred1.replace('___', ': ')[:28]}\n{r.conf1:.2f} {r.status[:4]}" + (f"\n(true: {r.true.split('___')[-1][:20]})" if r.true else ""), fontsize=6, color=col)
    fig.tight_layout(); fig.savefig(rep / "photo_check.png", dpi=110)
    print(f"\nDetails: {rep/'photo_check.csv'}   Contact sheet (errors first): {rep/'photo_check.png'}")


if __name__ == "__main__":
    main()
