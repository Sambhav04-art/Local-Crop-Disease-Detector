"""STEP 3: test-set metrics, calibration, low-confidence threshold, stress tests, field test.
   python -m cropdx.evaluate [--model auto|name] [--smoke]"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .config import load_config, path_of
from .data import EvalTransform, ManifestDataset, subsample_idx
from .metrics import confusion, report_dict
from .models import build_model
from .perturb import PERTURBATIONS, REAL_BG_TESTS, SHORTCUT_TEST, Perturbation, list_images
from .train import describe_device, num_workers, pick_device


@torch.no_grad()
def logits_for(model, df, tf, device, workers):
    dl = DataLoader(ManifestDataset(df, tf), batch_size=128, num_workers=workers)
    model.eval()
    return torch.cat([model(x.to(device)).float().cpu() for x, _ in dl]).numpy()


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def fit_temperature(logits, y):
    z, t = torch.tensor(logits), torch.tensor(y)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)
    def closure():
        opt.zero_grad(); l = F.cross_entropy(z / logT.exp(), t); l.backward(); return l
    opt.step(closure)
    return float(logT.exp())


def ece(conf, correct, bins=15):
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def selective(conf, correct, thr):
    acc = conf >= thr
    return {"coverage": float(acc.mean()),
            "selective_accuracy": float(correct[acc].mean()) if acc.any() else float("nan"),
            "wrong_but_accepted": float(((~correct) & acc).mean())}


def choose_threshold(conf, correct, target, lo, hi):
    rows = []
    for t in np.round(np.arange(0.30, 1.0, 0.01), 2):
        rows.append({"threshold": float(t), **selective(conf, correct, t)})
    tab = pd.DataFrame(rows)
    ok = tab[(tab.selective_accuracy >= target) & (tab.threshold >= lo)]
    thr = float(ok.threshold.min()) if len(ok) else float(hi)
    return min(max(thr, lo), hi), tab


def pick_best(runs):
    best = None
    for f in runs.glob("*/val_metrics.json"):
        m = json.loads(f.read_text())
        key = (m["val_macro_f1"], -m["params_millions"])
        if best is None or key > best[0]:
            best = (key, m["model"])
    if not best:
        raise SystemExit("No trained models in runs/. Run training first.")
    return best[1]


def plots(cm, names, tab, conf, correct, rep):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(14, 12)); im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6); ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Normalised confusion matrix (test set)")
    fig.colorbar(im, fraction=0.03); fig.tight_layout(); fig.savefig(rep / "confusion_matrix.png", dpi=130); plt.close(fig)

    edges = np.linspace(0, 1, 11); xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 5: xs.append(conf[m].mean()); ys.append(correct[m].mean())
    fig, ax = plt.subplots(figsize=(4.5, 4.5)); ax.plot([0, 1], [0, 1], "k--"); ax.plot(xs, ys, "o-")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.set_title("Reliability (calibrated, test)")
    fig.tight_layout(); fig.savefig(rep / "reliability_diagram.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(tab.threshold, tab.selective_accuracy, label="accuracy of accepted")
    ax.plot(tab.threshold, tab.coverage, label="coverage (share accepted)"); ax.legend(); ax.set_xlabel("confidence threshold")
    ax.set_title("Threshold analysis (validation, clean + stressed)"); fig.tight_layout()
    fig.savefig(rep / "threshold_analysis.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config"); ap.add_argument("--model", default="auto"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config); cal = cfg["calibration"]
    rep = path_of(cfg, "reports_dir"); rep.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path_of(cfg, "processed_dir") / "manifest.csv")
    classes = json.loads((path_of(cfg, "processed_dir") / "classes.json").read_text())
    ids = [c["classId"] for c in classes]; names = [c["displayName"] for c in classes]
    name = pick_best(path_of(cfg, "runs_dir")) if a.model == "auto" else a.model
    print("Evaluating model:", name)
    device, workers, size = pick_device(), (0 if a.smoke else num_workers(cfg)), cfg["image"]["size"]
    ck = torch.load(path_of(cfg, "runs_dir") / name / "best.pt", map_location="cpu")
    model = build_model(name, len(ids), pretrained=False); model.load_state_dict(ck["state_dict"]); model.to(device).eval()

    val, test = df[df.split == "val"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)
    if a.smoke:
        val, test = val.iloc[subsample_idx(val, 15)].reset_index(drop=True), test.iloc[subsample_idx(test, 15)].reset_index(drop=True)
    ev = EvalTransform(size)
    lv, lt = logits_for(model, val, ev, device, workers), logits_for(model, test, ev, device, workers)
    yv, yt = val.class_idx.values, test.class_idx.values

    # ---- calibration
    T_ = fit_temperature(lv, yv)
    pt = softmax(lt / T_); pred = pt.argmax(1); conf = pt.max(1); correct = pred == yt
    raw_conf = softmax(lt).max(1)
    ece_raw, ece_cal = ece(raw_conf, correct), ece(conf, correct)

    # ---- class-wise metrics + confusion matrix
    labels = list(range(len(ids)))
    r = report_dict(yt, pred, names)
    pd.DataFrame(r).T.to_csv(rep / "per_class_metrics.csv")
    cm = confusion(yt, pred, len(ids))
    pd.DataFrame(cm, index=names, columns=names).to_csv(rep / "confusion_matrix.csv")
    off = [(names[i], names[j], int(cm[i, j])) for i in labels for j in labels if i != j and cm[i, j] > 0]
    pd.DataFrame(sorted(off, key=lambda t: -t[2])[:15], columns=["true", "predicted", "count"]).to_csv(rep / "top_confusions.csv", index=False)
    top5 = float(np.mean([yt[i] in np.argsort(-pt[i])[:5] for i in range(len(yt))]))

    # ---- stress tests (validation subset -> threshold, test subset -> reporting)
    n_sub = 5 if a.smoke else cal["robustness_per_class"]
    iv, it = subsample_idx(val, n_sub, 1), subsample_idx(test, n_sub, 2)
    sv, st = val.iloc[iv].reset_index(drop=True), test.iloc[it].reset_index(drop=True)
    stress = {"clean": (lv[iv], lt[it])}
    bgp = list_images(path_of(cfg, "backgrounds_test_dir", "data/backgrounds_test"))
    for p in PERTURBATIONS:
        if p in REAL_BG_TESTS and not bgp:
            continue                      # needs held-out real background photos in data/backgrounds_test/
        tf = T.Compose([Perturbation(p, bgp), ev]); print("stress test:", p)
        stress[p] = (logits_for(model, sv, tf, device, workers), logits_for(model, st, tf, device, workers))
    pool_c, pool_ok = [], []
    for k, (lvv, _) in stress.items():
        pp = softmax(lvv / T_); pool_c.append(pp.max(1)); pool_ok.append(pp.argmax(1) == sv.class_idx.values)
    thr, tab = choose_threshold(np.concatenate(pool_c), np.concatenate(pool_ok), cal["target_selective_accuracy"],
                                cal["min_threshold"], cal["max_threshold"])
    tab.to_csv(rep / "threshold_analysis.csv", index=False)
    rob = []
    for k, (_, ltt) in stress.items():
        pp = softmax(ltt / T_); c = pp.argmax(1) == st.class_idx.values
        rob.append({"condition": k, "accuracy": float(c.mean()), "mean_confidence": float(pp.max(1).mean()),
                    **selective(pp.max(1), c, thr)})
    pd.DataFrame(rob).to_csv(rep / "robustness.csv", index=False)

    # ---- shortcut / background-bias test: leaf painted out, only background left
    tfc = T.Compose([Perturbation(SHORTCUT_TEST), ev])
    pc = softmax(logits_for(model, st, tfc, device, workers) / T_)
    shortcut = {"accuracy_without_leaf": float((pc.argmax(1) == st.class_idx.values).mean()),
                "chance_level": 1.0 / len(ids), "mean_confidence": float(pc.max(1).mean()),
                "share_wrongly_accepted_above_threshold": float((pc.max(1) >= thr).mean())}
    shortcut["verdict"] = ("BACKGROUND SHORTCUT: model can guess the class without seeing the leaf"
                           if shortcut["accuracy_without_leaf"] > 4 * shortcut["chance_level"] else "ok (close to chance)")
    (rep / "shortcut_test.json").write_text(json.dumps(shortcut, indent=2)); print("shortcut test:", shortcut)

    # ---- optional real-world field test
    field = None
    fdir = path_of(cfg, "field_test_dir")
    items = [(f, d.name) for d in sorted(fdir.glob("*")) if d.is_dir() for f in d.iterdir()
             if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")] if fdir.exists() else []
    if items:
        fdf = pd.DataFrame({"path": [str(f) for f, _ in items], "class_idx": [ids.index(c) if c in ids else -1 for _, c in items]})
        pf = softmax(logits_for(model, fdf, ev, device, 0) / T_); known = fdf.class_idx.values >= 0
        field = {"n_images": len(fdf), "n_known": int(known.sum()), "n_out_of_scope": int((~known).sum())}
        if known.any():
            ok = pf.argmax(1)[known] == fdf.class_idx.values[known]
            field["known_accuracy"] = float(ok.mean())
            if len(ok) >= 20:
                field["suggested_threshold_for_field_photos"] = choose_threshold(pf.max(1)[known], ok, 0.90, 0.5, 0.99)[0]
            field.update({f"known_{k}": v for k, v in selective(pf.max(1)[known], ok, thr).items()})
        if (~known).any():
            field["out_of_scope_rejected_share"] = float((pf.max(1)[~known] < thr).mean())
        (rep / "field_test.json").write_text(json.dumps(field, indent=2))

    plots(cm, names, tab, conf, correct, rep)
    clean_sel = selective(conf, correct, thr)
    summary = {"model": name, "temperature": T_, "confidence_threshold": thr, "test_images": int(len(yt)),
               "test_accuracy": float(correct.mean()), "test_top5_accuracy": top5,
               "test_macro_precision": r["macro avg"]["precision"], "test_macro_recall": r["macro avg"]["recall"],
               "test_macro_f1": r["macro avg"]["f1-score"], "test_weighted_f1": r["weighted avg"]["f1-score"],
               "ece_before_calibration": ece_raw, "ece_after_calibration": ece_cal,
               "clean_test_at_threshold": clean_sel, "field_test": field, "shortcut_test": shortcut, "smoke_run": a.smoke}
    (rep / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    (rep / "deployment_config.json").write_text(json.dumps({"model": name, "temperature": T_, "confidence_threshold": thr,
        "test_accuracy": summary["test_accuracy"], "test_macro_f1": summary["test_macro_f1"]}, indent=2))
    print(json.dumps(summary, indent=2)); print(pd.DataFrame(rob).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
