"""Map a REAL-photo dataset (e.g. PlantDoc) with its own folder names onto your class ids and copy it into
data/field_train and data/field_test.

  python -m cropdx.import_field_dataset --src "D:\\PlantDoc\\train" "D:\\PlantDoc\\test"          (dry run: shows the mapping)
  python -m cropdx.import_field_dataset --src "D:\\PlantDoc\\train" "D:\\PlantDoc\\test" --yes    (copies files)
CHECK THE PRINTED MAPPING before using --yes: fuzzy matching can be wrong."""
import argparse, json, random, re, shutil
from pathlib import Path

from .config import load_config, path_of
from .labels import PLANTVILLAGE_38, parse_class_name

STOP = {"leaf", "leaves", "the", "of", "plant", "and", "with", "a", "an", "including"}
SYN = {"maize": "corn", "mould": "mold", "mites": "mite", "spots": "spot", "blights": "blight", "viruses": "virus", "bell": "pepper"}
EXT = (".jpg", ".jpeg", ".png", ".webp")


def toks(s):
    out = set()
    for w in re.findall(r"[a-z]+", re.sub(r"\(.*?\)", " ", s.lower())):
        w = SYN.get(w, w)
        if w not in STOP:
            out.add(w)
    return out


def build_index(class_ids):
    idx = []
    for cid in class_ids:
        info = parse_class_name(cid); crop = next(iter(sorted(toks(info["crop"]))), "")
        crop = sorted(toks(info["crop"]), key=lambda w: info["crop"].lower().find(w))[0] if toks(info["crop"]) else crop
        idx.append((cid, crop, toks(info["condition"]) - {crop}, info["healthy"]))
    return idx


def map_folder(name, idx):
    S = toks(name)
    best = None
    for cid, crop, D, healthy in idx:
        if crop not in S:
            continue
        rest = S - {crop}
        if healthy:
            score = 1.0 if (not rest or "healthy" in rest) else 0.0
        else:
            score = len(D & rest) / len(D) if D else 0.0
        if score >= 0.5 and (best is None or score > best[1]):
            best = (cid, score)
        elif best and score == best[1] and score >= 0.5 and cid != best[0]:
            best = (best[0], score - 0.01)  # tie => ambiguous, keep first but lower confidence flag
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--test-fraction", type=float, default=0.3); ap.add_argument("--yes", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); cfg = load_config()
    cj = path_of(cfg, "processed_dir") / "classes.json"
    ids = [c["classId"] for c in json.loads(cj.read_text())] if cj.exists() else PLANTVILLAGE_38
    idx = build_index(ids); found = {}
    for src in a.src:
        for d in sorted(x for x in Path(src).iterdir() if x.is_dir()):
            files = [f for f in d.iterdir() if f.suffix.lower() in EXT]
            m = map_folder(d.name, idx)
            print(f"{d.name!r:55} -> {m[0] if m else '--- NOT MAPPED (skipped)':50} {len(files):5d} images" + ("  (ambiguous!)" if m and m[1] < 1 and m[1] != 0.5 else ""))
            if m:
                found.setdefault(m[0], []).extend(files)
    print(f"\nMapped {len(found)} of {len(ids)} classes; {sum(map(len, found.values()))} images.")
    if not a.yes:
        print("Dry run. Re-run with --yes after you checked the mapping."); return
    rnd = random.Random(a.seed); ftrain, ftest = path_of(cfg, "field_train_dir", "data/field_train"), path_of(cfg, "field_test_dir")
    n_tr = n_te = 0
    for cid, files in found.items():
        files = sorted(files); rnd.shuffle(files)
        k = int(len(files) * a.test_fraction) if len(files) >= 10 else 0
        for i, f in enumerate(files):
            dst = (ftest if i < k else ftrain) / cid; dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst / f"{f.parent.parent.name}_{f.parent.name}_{f.name}"); n_te += i < k; n_tr += i >= k
    print(f"Copied {n_tr} images to {ftrain} and {n_te} to {ftest}.")
    print("NOTE: a random split can put near-identical photos on both sides (web-scraped sets contain duplicates), so the field-test score may be optimistic.")


if __name__ == "__main__":
    main()
