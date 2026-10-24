"""STEP 1: find the dataset, clean it, remove duplicates, inspect balance, leakage-safe split.

Usage:  python -m cropdx.prepare_data            (auto-detects the 'color' folder under data/raw)
        python -m cropdx.prepare_data --data-dir "D:/plantvillage dataset/color"
"""
from __future__ import annotations
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .config import load_config, path_of
from .dedupe import UnionFind, fingerprint, near_pairs
from .labels import parse_class_name

SPLIT_NAMES = {"train", "training", "valid", "val", "validation", "test"}
BAD_VARIANT = re.compile(r"gr[ae]y|segment|mask|annot|bbox", re.I)


# ----------------------------------------------------------------- discovery
def find_candidate_dirs(root: Path, exts, max_depth=4):
    """Directories whose sub-folders look like class folders full of images."""
    out = []
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "__"))]
        d = Path(dirpath)
        if len(d.relative_to(root).parts) >= max_depth:
            dirnames[:] = []
        if len(dirnames) < 2:
            continue
        n_img = 0
        for sub in dirnames:
            try:
                with os.scandir(d / sub) as it:
                    if any(e.is_file() and Path(e.name).suffix.lower() in exts for e in it):
                        n_img += 1
            except OSError:
                pass
        if n_img >= max(2, int(0.8 * len(dirnames))):
            out.append(d)
    return out


def select_dirs(cands, root: Path, prefer_regex: str):
    if not cands:
        raise SystemExit(f"No class-folder dataset found under {root}. Extract the Kaggle zip into that folder.")
    parts = lambda c: c.relative_to(root).parts or (c.name,)
    pref = re.compile(prefer_regex, re.I)
    good = [c for c in cands if any(pref.search(p) for p in parts(c))]
    if not good:
        good = [c for c in cands if not any(BAD_VARIANT.search(p) for p in parts(c))]
    if len(good) > 1 and all(c.name.lower() in SPLIT_NAMES for c in good):
        return good  # train/valid/test layout: merged, then re-split without leakage
    if len(good) > 1:  # e.g. the zip contains a duplicate nested copy: take the shallowest
        good.sort(key=lambda c: len(parts(c)))
        if len(parts(good[0])) < len(parts(good[1])):
            return [good[0]]
        raise SystemExit("Several candidate dataset folders found:\n  " + "\n  ".join(map(str, good)) +
                         "\nRe-run with --data-dir <the folder that contains the class folders>.")
    return good


def scan_images(dirs, exts, crops, exclude):
    rows = []
    for d in dirs:
        for cls in sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith((".", "__"))):
            if cls.name in exclude:
                continue
            crop = parse_class_name(cls.name)["crop"].lower()
            if crops and not any(crop.startswith(c.lower()) for c in crops):
                continue
            for f in sorted(cls.iterdir()):
                if f.is_file() and f.suffix.lower() in exts:
                    rows.append((cls.name, str(f.resolve())))
    return pd.DataFrame(rows, columns=["class_name", "path"])


# ------------------------------------------------------------------ cleaning
def analyze(path, c):
    try:
        with Image.open(path) as im:
            w, h = im.size
            if im.format == "JPEG":
                im.draft("L", (160, 160))
            g = im.convert("L")
    except Exception:
        return {"ok": False, "reason": "corrupt_or_unreadable"}
    if min(w, h) < c["min_side_px"]:
        return {"ok": False, "reason": "too_small"}
    if max(w, h) / max(1, min(w, h)) > c["max_aspect_ratio"]:
        return {"ok": False, "reason": "extreme_aspect_ratio"}
    arr = np.asarray(g, dtype=np.float32)
    mean, std = float(arr.mean()), float(arr.std())
    if std < c["blank_std_threshold"]:
        return {"ok": False, "reason": "blank_or_uniform"}
    if mean < c["min_mean_brightness"]:
        return {"ok": False, "reason": "almost_black"}
    if mean > c["max_mean_brightness"]:
        return {"ok": False, "reason": "almost_white"}
    thumb, hashes = fingerprint(g)
    return {"ok": True, "reason": "", "width": w, "height": h, "thumb": thumb, "hashes": hashes}


# --------------------------------------------------------------------- split
def grouped_split(df, test_frac, val_frac, seed):
    from .metrics import grouped_stratified_split
    ids = grouped_stratified_split(df.group.values, df.class_idx.values,
                                   [1 - test_frac - val_frac, val_frac, test_frac], seed)
    return np.array(["train", "val", "test"], dtype=object)[ids]


def plots(before, after, df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = sorted(after.index)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(names))))
    y = np.arange(len(names))
    ax.barh(y + 0.2, [before.get(n, 0) for n in names], 0.4, label="scanned")
    ax.barh(y - 0.2, [after[n] for n in names], 0.4, label="after cleaning")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("images"); ax.legend(); ax.set_title("Class distribution")
    fig.tight_layout(); fig.savefig(out_dir / "class_distribution.png", dpi=130); plt.close(fig)

    firsts = df[df.split == "train"].groupby("class_idx").head(1).sort_values("class_idx")
    n = len(firsts); cols = 8; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (_, r) in zip(np.atleast_1d(axes).ravel(), firsts.iterrows()):
        ax.imshow(Image.open(r.path).convert("RGB").resize((160, 160)))
        ax.set_title(r.class_name[:26], fontsize=5)
    fig.tight_layout(); fig.savefig(out_dir / "sample_grid.png", dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--data-dir", help="folder that directly contains the class folders")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds, cl = cfg["dataset"], cfg["cleaning"]
    exts = set(ds["extensions"])
    root = path_of(cfg, "data_root")
    out, rep = path_of(cfg, "processed_dir"), path_of(cfg, "reports_dir")
    out.mkdir(parents=True, exist_ok=True); rep.mkdir(parents=True, exist_ok=True)

    if args.data_dir:
        dirs = [Path(args.data_dir)]
    else:
        dirs = select_dirs(find_candidate_dirs(root, exts), root, ds["prefer_folder_regex"])
    print("Using dataset folder(s):"); [print("  ", d) for d in dirs]

    df = scan_images(dirs, exts, ds["supported_crops"], set(ds["exclude_classes"]))
    if df.empty:
        raise SystemExit("No images found (check supported_crops / exclude_classes / folder).")
    scanned = df.class_name.value_counts().to_dict()
    print(f"Scanned {len(df)} images in {df.class_name.nunique()} class folders. Analysing...")

    with ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 4) * 2)) as ex:
        res = list(tqdm(ex.map(lambda p: analyze(p, cl), df.path), total=len(df)))
    df["ok"] = [r["ok"] for r in res]
    df["reason"] = [r["reason"] for r in res]
    removed = df[~df.ok][["path", "class_name", "reason"]].copy()
    keep = np.nonzero(df.ok.values)[0]
    df = df.iloc[keep].reset_index(drop=True)
    res = [res[i] for i in keep]
    df["width"] = [r["width"] for r in res]; df["height"] = [r["height"] for r in res]
    thumbs = np.stack([r["thumb"] for r in res]); hashes = np.stack([r["hashes"] for r in res])

    # ---- duplicates / label conflicts / similarity groups
    print("Searching near-duplicates (flip/rotation invariant)...")
    a, b, mad = near_pairs(hashes, thumbs, cl["max_hamming"])
    uf_dup, n = UnionFind(len(df)), len(df)
    for i, j in zip(a[mad <= cl["duplicate_mad"]], b[mad <= cl["duplicate_mad"]]):
        uf_dup.union(int(i), int(j))
    clusters = {}
    for i in range(n):
        clusters.setdefault(uf_dup.find(i), []).append(i)
    drop, why = set(), {}
    for members in clusters.values():
        if len(members) == 1:
            continue
        if df.class_name.iloc[members].nunique() > 1:
            for m in members:
                drop.add(m); why[m] = "cross_class_duplicate_label_conflict"
        else:
            for m in members[1:]:
                drop.add(m); why[m] = "duplicate_of_kept_image"
    dd = df.iloc[sorted(drop)][["path", "class_name"]].copy()
    dd["reason"] = [why[i] for i in sorted(drop)]
    removed = pd.concat([removed, dd], ignore_index=True)
    alive = np.array([i not in drop for i in range(n)])

    uf_grp = UnionFind(n)
    sel = (mad <= cl["group_mad"]) & alive[a] & alive[b]
    for i, j in zip(a[sel], b[sel]):
        uf_grp.union(int(i), int(j))
    df["group"] = [uf_grp.find(i) for i in range(n)]
    df = df[alive].reset_index(drop=True)

    # ---- class filtering & indices
    counts = df.class_name.value_counts()
    small = counts[counts < ds["min_images_per_class"]]
    if len(small):
        print("Dropping classes with too few clean images:", dict(small))
        df = df[~df.class_name.isin(small.index)].reset_index(drop=True)
    classes = sorted(df.class_name.unique())
    cidx = {c: i for i, c in enumerate(classes)}
    df["class_idx"] = df.class_name.map(cidx)

    # ---- leakage-safe split
    df["split"] = grouped_split(df, cfg["split"]["test"], cfg["split"]["val"], cfg["split"]["seed"])
    g = df.groupby("group").split.nunique()
    assert (g == 1).all(), "LEAKAGE: a similarity group spans several splits"
    for s in ("train", "val", "test"):
        missing = set(range(len(classes))) - set(df[df.split == s].class_idx)
        assert not missing, f"split {s} misses classes {missing}"

    df["source"] = "plantvillage"
    ft = path_of(cfg, "field_train_dir", "data/field_train")
    if ft.is_dir():
        rows = []
        for d in sorted(x for x in ft.iterdir() if x.is_dir()):
            if d.name not in cidx:
                print(f"  field_train: folder '{d.name}' is not a supported class id - skipped"); continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in exts:
                    try:
                        with Image.open(f) as im:
                            im.verify(); w, h = im.size
                    except Exception:
                        continue
                    rows.append((str(f.resolve()), d.name, cidx[d.name], w, h))
        if rows:
            fdf = pd.DataFrame(rows, columns=["path", "class_name", "class_idx", "width", "height"])
            fdf["group"] = -1 - np.arange(len(fdf)); fdf["split"] = "train"; fdf["source"] = "field"
            df = pd.concat([df, fdf], ignore_index=True)
            print(f"Added {len(fdf)} real field photos from {ft} to the TRAIN split "
                  f"({fdf.class_name.nunique()} classes). Keep held-out photos in data/field_test/.")
    df[["path", "class_name", "class_idx", "group", "split", "width", "height", "source"]].to_csv(out / "manifest.csv", index=False)
    (out / "classes.json").write_text(json.dumps([{"index": i, **parse_class_name(c)} for i, c in enumerate(classes)],
                                                 indent=2), encoding="utf-8")
    removed.to_csv(rep / "removed_images.csv", index=False)
    per = df.groupby(["class_name", "split"]).size().unstack(fill_value=0)
    per.to_csv(rep / "split_per_class.csv")
    after = df.class_name.value_counts()
    report = {
        "dataset_dirs": [str(d) for d in dirs], "num_classes": len(classes), "scanned_images": int(sum(scanned.values())),
        "removed_by_reason": removed.reason.value_counts().to_dict(), "clean_images": int(len(df)),
        "classes_dropped_too_small": {k: int(v) for k, v in small.items()},
        "imbalance_ratio_max_over_min": round(float(after.max() / after.min()), 2),
        "smallest_class": [after.idxmin(), int(after.min())], "largest_class": [after.idxmax(), int(after.max())],
        "split_images": df.split.value_counts().to_dict(), "field_train_images": int((df.source == "field").sum()), "similarity_groups": int(df.group.nunique()),
        "largest_group": int(df.group.value_counts().max()),
        "image_size_top5": df.assign(s=df.width.astype(str) + "x" + df.height.astype(str)).s.value_counts().head(5).to_dict(),
    }
    (rep / "data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plots(scanned, after, df, rep)
    print(json.dumps(report, indent=2))
    print(f"\nDone. Manifest: {out/'manifest.csv'}")


if __name__ == "__main__":
    main()
