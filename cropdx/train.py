"""STEP 2: train + compare models.   python -m cropdx.train [--models a b] [--epochs N] [--smoke]"""
from __future__ import annotations
import argparse, json, os, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config, path_of
from .data import EvalTransform, ManifestDataset, build_train_transform, subsample_idx
from .metrics import macro_f1
from .models import build_model, n_params


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device):
    if device.type == "cuda":
        print(f"device: cuda - {torch.cuda.get_device_name(0)} (torch {torch.__version__})")
    else:
        print(f"device: {device.type}  (torch {torch.__version__})")
        print("\n" + "!" * 70 + "\n WARNING: training on the CPU. This is 10-30x slower than on the GPU.\n"
              " If you own an NVIDIA GPU you probably installed the CPU-only PyTorch.\n"
              " Run:  python -m cropdx.diagnose   and follow its instructions.\n" + "!" * 70 + "\n")


def num_workers(cfg):
    w = cfg["train"]["num_workers"]
    return max(2, min(8, (os.cpu_count() or 4) - 1)) if w == "auto" else int(w)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); lg, ys = [], []
    for x, y in loader:
        lg.append(model(x.to(device)).float().cpu()); ys.append(y)
    return torch.cat(lg), torch.cat(ys)


def cpu_latency_ms(model, size):
    model = model.cpu().eval(); x = torch.randn(1, 3, size, size)
    with torch.no_grad():
        for _ in range(3): model(x)
        t = []
        for _ in range(15):
            s = time.perf_counter(); model(x); t.append((time.perf_counter() - s) * 1000)
    return float(np.median(t))


def run_one(name, cfg, df, classes, device, epochs, limit):
    tc, size = cfg["train"], cfg["image"]["size"]
    seed_all(tc["seed"])
    run_dir = path_of(cfg, "runs_dir") / name; run_dir.mkdir(parents=True, exist_ok=True)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    if limit:
        tr, va = tr.iloc[subsample_idx(tr, limit)], va.iloc[subsample_idx(va, max(5, limit // 3))]
    k = int(tc.get("field_oversample", 5))
    if not limit and k > 1 and "source" in tr.columns and (tr.source == "field").any():
        tr = pd.concat([tr] + [tr[tr.source == "field"]] * (k - 1), ignore_index=True)
        print(f"Real field photos in training: {(tr.source == 'field').sum()} (oversampled x{k})")
    workers = 0 if limit else num_workers(cfg)
    dl_tr = DataLoader(ManifestDataset(tr, build_train_transform(cfg)), batch_size=tc["batch_size"], shuffle=True,
                       num_workers=workers, drop_last=True, pin_memory=device.type == "cuda",
                       persistent_workers=workers > 0)
    dl_va = DataLoader(ManifestDataset(va, EvalTransform(size)), batch_size=128, num_workers=workers)

    cnt = np.bincount(tr.class_idx, minlength=len(classes)).astype(np.float64).clip(min=1)
    w = (cnt.mean() / cnt) ** 0.5 if tc["class_weighting"] == "sqrt" else np.ones_like(cnt)
    weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=tc["label_smoothing"])

    try:
        model = build_model(name, len(classes), tc["pretrained"]).to(device)
    except Exception as e:
        raise SystemExit(f"Could not build {name} (pretrained download failed? set train.pretrained: false): {e}")
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc["lr"], total_steps=epochs * len(dl_tr), pct_start=0.15)
    amp = device.type == "cuda"
    if amp:
        model = model.to(memory_format=torch.channels_last)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best, bad, hist, t0 = -1.0, 0, [], time.time()
    for ep in range(1, epochs + 1):
        model.train(); run = 0.0; t_ep = time.time()
        for x, y in tqdm(dl_tr, desc=f"{name} ep{ep}/{epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if amp:
                x = x.contiguous(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
        lg, y = predict(model, dl_va, device)
        pred = lg.argmax(1).numpy()
        acc, f1 = float((pred == y.numpy()).mean()), macro_f1(y.numpy(), pred, len(classes))
        hist.append({"epoch": ep, "train_loss": run / len(dl_tr), "val_acc": acc, "val_macro_f1": f1})
        print(f"[{name}] epoch {ep}: loss {run/len(dl_tr):.4f} val_acc {acc:.4f} val_macroF1 {f1:.4f}  ({time.time()-t_ep:.0f}s, {len(tr)/(time.time()-t_ep):.0f} img/s)")
        if f1 > best:
            best, bad = f1, 0
            torch.save({"model": name, "state_dict": model.state_dict(), "classes": classes, "image_size": size,
                        "epoch": ep}, run_dir / "best.pt")
        else:
            bad += 1
            if bad >= tc["patience"]:
                print("early stopping"); break
    pd.DataFrame(hist).to_csv(run_dir / "history.csv", index=False)
    b = max(hist, key=lambda h: h["val_macro_f1"])
    metrics = {"model": name, "val_acc": b["val_acc"], "val_macro_f1": b["val_macro_f1"], "best_epoch": b["epoch"],
               "epochs_run": len(hist), "params_millions": round(n_params(model) / 1e6, 2),
               "train_minutes": round((time.time() - t0) / 60, 1), "cpu_latency_ms": round(cpu_latency_ms(model, size), 1)}
    (run_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config"); ap.add_argument("--models", nargs="+"); ap.add_argument("--epochs", type=int)
    ap.add_argument("--smoke", action="store_true", help="tiny quick run to verify the pipeline works")
    a = ap.parse_args()
    cfg = load_config(a.config)
    df = pd.read_csv(path_of(cfg, "processed_dir") / "manifest.csv")
    classes = [c["classId"] for c in json.loads((path_of(cfg, "processed_dir") / "classes.json").read_text())]
    device = pick_device(); describe_device(device)
    epochs = 1 if a.smoke else (a.epochs or cfg["train"]["epochs"])
    rows = [run_one(m, cfg, df, classes, device, epochs, 20 if a.smoke else 0) for m in (a.models or cfg["train"]["models"])]
    comp = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    out = path_of(cfg, "reports_dir"); out.mkdir(parents=True, exist_ok=True)
    comp.to_csv(out / "model_comparison.csv", index=False)
    print("\n", comp.to_string(index=False))


if __name__ == "__main__":
    main()
