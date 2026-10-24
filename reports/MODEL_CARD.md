# Model Card - Crop Disease Classifier v1.0.0
**Intended use:** AI-assisted indication of a likely leaf disease/healthy state from a photo of a single leaf, to support (not replace) an expert.
**Not for:** guaranteed diagnosis, treatment/dosage decisions, pest or nutrient-deficiency identification, non-leaf images, unsupported crops.
> AI-assisted indication only. This is not a guaranteed agricultural diagnosis. Confirm with a qualified agronomist or local agricultural extension service before taking any treatment decision.

* Architecture: efficientnet_b0 (ImageNet pre-trained where applicable), input 224x224 RGB, exported to ONNX.
* Training data: PlantVillage, 54272 cleaned images, 38 classes (crop___condition).
* Test accuracy 0.997, macro-F1 0.996 (lab-condition images).
* Output: top-k classes with calibrated confidence; `LOW_CONFIDENCE` if top confidence < 0.70.
* Supported classes: see `artifacts/labels.json` / `GET /v1/classes`.

## Known limitations (read before relying on any output)
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
