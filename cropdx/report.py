"""STEP 5: build reports/EVALUATION_REPORT.md and reports/MODEL_CARD.md from real results."""
import json
import pandas as pd
from .config import load_config, path_of
from .constants import DISCLAIMER


def md(df, nd=3):
    df = df.round(nd)
    h = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return h + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.values) + "\n"


LIMITS = """## Known limitations (read before relying on any output)
1. **Lab images.** PlantVillage photos are single leaves on plain backgrounds under controlled light. Real field photos
   (soil, hands, several leaves, shadows, wet leaves, partial views) are a different distribution. Accuracy on the
   clean test set is **not** the accuracy you will see in the field; the stress tests below only approximate that gap.
2. **Possible optimistic test score.** PlantVillage has no leaf/plant IDs. We remove flip/rotation/near duplicates and
   keep similar images in the same split, but different photos of the *same leaf* may still straddle splits.
3. **Closed set.** The model only knows the listed classes. It will still try to name one of them for a non-leaf image,
   an unsupported crop, or a disease outside the list; the confidence threshold reduces but cannot eliminate this.
4. **Single dominant condition.** Plants with two simultaneous problems (disease + nutrient deficiency + pests) are not handled.
5. **Leaf only.** Fruit, stem, root and whole-plant images are out of scope.
6. **Early-stage / look-alike conditions** (e.g. several tomato leaf spots, nutrient deficiency vs virus) are hard even for experts.
7. **Confidence is calibrated on PlantVillage-like data.** Re-validate the threshold with real field images before production.
8. No severity estimate and no treatment advice is produced.
"""


def main():
    cfg = load_config(); rep = path_of(cfg, "reports_dir")
    S = json.loads((rep / "metrics_summary.json").read_text())
    D = json.loads((rep / "data_report.json").read_text())
    cmp_ = pd.read_csv(rep / "model_comparison.csv") if (rep / "model_comparison.csv").exists() else None
    per = pd.read_csv(rep / "per_class_metrics.csv", index_col=0)
    rob = pd.read_csv(rep / "robustness.csv"); conf = pd.read_csv(rep / "top_confusions.csv")
    cls = per.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")[["precision", "recall", "f1-score", "support"]].reset_index()
    cls.columns = ["class", "precision", "recall", "f1", "support"]
    clean = S["clean_test_at_threshold"]
    t = f"""# Evaluation Report - Crop Disease Image Classifier
> {DISCLAIMER}

{'**NOTE: this report comes from a --smoke run (tiny subset); numbers are meaningless.**' if S.get('smoke_run') else ''}

## 1. Dataset & cleaning
* Source: PlantVillage (Kaggle), colour variant. Folders used: `{D['dataset_dirs']}`
* Classes supported: **{D['num_classes']}** (see `classes.json`); scanned images: {D['scanned_images']}; clean images kept: {D['clean_images']}
* Removed (reason: count): `{D['removed_by_reason']}`
* Class balance: largest `{D['largest_class']}`, smallest `{D['smallest_class']}`, imbalance ratio {D['imbalance_ratio_max_over_min']}x
  -> handled with sqrt-inverse-frequency class weights in the loss and macro-averaged metrics. Plot: `class_distribution.png`
* Split (grouped, stratified, seed 42): `{D['split_images']}`. {D['similarity_groups']} similarity groups; a group never spans two splits (asserted in code).

## 2. Preprocessing & augmentation
Inference: EXIF-rotate -> RGB -> resize 224x224 bilinear -> ImageNet mean/std. Identical code in training-eval and API (`cropdx/preprocess.py`).
Training augmentation: random synthetic background (30%), JPEG compression q15-60 (30%), random resized crop, H/V flips, rotation +-30,
colour jitter (small hue), blur, random erasing.

## 3. Model comparison (validation set)
{md(cmp_) if cmp_ is not None else 'n/a'}
Selected model: **{S['model']}** (highest validation macro-F1; ties -> fewer parameters).

## 4. Test-set results ({S['test_images']} images, never used for training or model selection)
| metric | value |
|---|---|
| accuracy | {S['test_accuracy']:.4f} |
| top-5 accuracy | {S['test_top5_accuracy']:.4f} |
| macro precision / recall / F1 | {S['test_macro_precision']:.4f} / {S['test_macro_recall']:.4f} / {S['test_macro_f1']:.4f} |
| weighted F1 | {S['test_weighted_f1']:.4f} |
| ECE before / after temperature scaling (T={S['temperature']:.3f}) | {S['ece_before_calibration']:.4f} / {S['ece_after_calibration']:.4f} |

Class-wise metrics (`per_class_metrics.csv`), confusion matrix (`confusion_matrix.png` / `.csv`):
{md(cls)}
Most frequent confusions:
{md(conf)}
## 5. Confidence & low-confidence threshold
Confidence = max softmax probability after temperature scaling (fitted on the validation set).
Threshold **{S['confidence_threshold']:.2f}** was chosen on validation images (clean + {len(rob)-1} simulated difficult conditions) as the lowest value
(>= {cfg['calibration']['min_threshold']}) where accepted predictions are >= {cfg['calibration']['target_selective_accuracy']:.0%} accurate.
Clean test at this threshold: coverage {clean['coverage']:.3f}, accuracy of accepted {clean['selective_accuracy']:.3f}.
Below the threshold the API returns `LOW_CONFIDENCE` and **no prediction**, only candidates. Plot: `threshold_analysis.png`.

## 6. Difficult conditions (stress tests on test subset)
Synthetic lighting / colour cast / blur / noise / low resolution / compression / close-up / background replacement. Background swaps are *procedural
approximations* and the model saw similar augmentation in training, so treat them as optimistic.
{md(rob)}
## 6b. Background-shortcut test (leaf painted out, only background left)
{json.dumps(S.get('shortcut_test'), indent=2) if S.get('shortcut_test') else 'not run'}
Accuracy far above chance (1/num_classes) means the model partly recognises PlantVillage backgrounds instead of leaves.

## 7. Real-world field test
{json.dumps(S['field_test'], indent=2) if S['field_test'] else 'No images supplied in data/field_test/. **Strongly recommended before production.**'}

{LIMITS}"""
    (rep / "EVALUATION_REPORT.md").write_text(t, encoding="utf-8")

    card = f"""# Model Card - Crop Disease Classifier v{cfg['export']['model_version']}
**Intended use:** AI-assisted indication of a likely leaf disease/healthy state from a photo of a single leaf, to support (not replace) an expert.
**Not for:** guaranteed diagnosis, treatment/dosage decisions, pest or nutrient-deficiency identification, non-leaf images, unsupported crops.
> {DISCLAIMER}

* Architecture: {S['model']} (ImageNet pre-trained where applicable), input 224x224 RGB, exported to ONNX.
* Training data: PlantVillage, {D['clean_images']} cleaned images, {D['num_classes']} classes (crop___condition).
* Test accuracy {S['test_accuracy']:.3f}, macro-F1 {S['test_macro_f1']:.3f} (lab-condition images).
* Output: top-k classes with calibrated confidence; `LOW_CONFIDENCE` if top confidence < {S['confidence_threshold']:.2f}.
* Supported classes: see `artifacts/labels.json` / `GET /v1/classes`.

{LIMITS}"""
    (rep / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    print("Wrote", rep / "EVALUATION_REPORT.md", "and MODEL_CARD.md")


if __name__ == "__main__":
    main()
