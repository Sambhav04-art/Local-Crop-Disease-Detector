# Evaluation Report - Crop Disease Image Classifier
> AI-assisted indication only. This is not a guaranteed agricultural diagnosis. Confirm with a qualified agronomist or local agricultural extension service before taking any treatment decision.



## 1. Dataset & cleaning
* Source: PlantVillage (Kaggle), colour variant. Folders used: `['C:\\Users\\sambh\\OneDrive\\Desktop\\coding\\team_task_gdg\\crop-disease-ai\\crop-disease-ai\\data\\raw\\plantvillage dataset\\color']`
* Classes supported: **38** (see `classes.json`); scanned images: 54305; clean images kept: 54272
* Removed (reason: count): `{'duplicate_of_kept_image': 33}`
* Class balance: largest `['Orange___Haunglongbing_(Citrus_greening)', 5502]`, smallest `['Potato___healthy', 152]`, imbalance ratio 36.2x
  -> handled with sqrt-inverse-frequency class weights in the loss and macro-averaged metrics. Plot: `class_distribution.png`
* Split (grouped, stratified, seed 42): `{'train': 37968, 'val': 8154, 'test': 8150}`. 50544 similarity groups; a group never spans two splits (asserted in code).

## 2. Preprocessing & augmentation
Inference: EXIF-rotate -> RGB -> resize 224x224 bilinear -> ImageNet mean/std. Identical code in training-eval and API (`cropdx/preprocess.py`).
Training augmentation: random synthetic background (30%), JPEG compression q15-60 (30%), random resized crop, H/V flips, rotation +-30,
colour jitter (small hue), blur, random erasing.

## 3. Model comparison (validation set)
| model | val_acc | val_macro_f1 | best_epoch | epochs_run | params_millions | train_minutes | cpu_latency_ms |
|---|---|---|---|---|---|---|---|
| efficientnet_b0 | 0.998 | 0.997 | 6 | 6 | 4.06 | 103.1 | 29.1 |
| mobilenet_v3_large | 0.997 | 0.996 | 5 | 6 | 4.25 | 62.5 | 16.6 |
| baseline_cnn | 0.945 | 0.932 | 6 | 6 | 0.99 | 56.8 | 20.0 |

Selected model: **efficientnet_b0** (highest validation macro-F1; ties -> fewer parameters).

## 4. Test-set results (8150 images, never used for training or model selection)
| metric | value |
|---|---|
| accuracy | 0.9971 |
| top-5 accuracy | 1.0000 |
| macro precision / recall / F1 | 0.9959 / 0.9969 / 0.9963 |
| weighted F1 | 0.9971 |
| ECE before / after temperature scaling (T=0.465) | 0.1453 / 0.0015 |

Class-wise metrics (`per_class_metrics.csv`), confusion matrix (`confusion_matrix.png` / `.csv`):
| class | precision | recall | f1 | support |
|---|---|---|---|---|
| Apple - Apple scab | 1.0 | 1.0 | 1.0 | 95.0 |
| Apple - Black rot | 1.0 | 1.0 | 1.0 | 93.0 |
| Apple - Cedar apple rust | 1.0 | 1.0 | 1.0 | 41.0 |
| Apple - healthy | 1.0 | 0.996 | 0.998 | 246.0 |
| Blueberry - healthy | 0.996 | 1.0 | 0.998 | 226.0 |
| Cherry (including sour) - Powdery mildew | 1.0 | 1.0 | 1.0 | 158.0 |
| Cherry (including sour) - healthy | 1.0 | 0.992 | 0.996 | 128.0 |
| Corn (maize) - Cercospora leaf spot Gray leaf spot | 0.916 | 0.987 | 0.95 | 77.0 |
| Corn (maize) - Common rust | 1.0 | 1.0 | 1.0 | 179.0 |
| Corn (maize) - Northern Leaf Blight | 0.993 | 0.953 | 0.972 | 148.0 |
| Corn (maize) - healthy | 1.0 | 1.0 | 1.0 | 174.0 |
| Grape - Black rot | 1.0 | 1.0 | 1.0 | 177.0 |
| Grape - Esca (Black Measles) | 1.0 | 1.0 | 1.0 | 208.0 |
| Grape - Leaf blight (Isariopsis Leaf Spot) | 1.0 | 1.0 | 1.0 | 162.0 |
| Grape - healthy | 1.0 | 1.0 | 1.0 | 64.0 |
| Orange - Haunglongbing (Citrus greening) | 1.0 | 1.0 | 1.0 | 825.0 |
| Peach - Bacterial spot | 1.0 | 1.0 | 1.0 | 345.0 |
| Peach - healthy | 1.0 | 1.0 | 1.0 | 54.0 |
| Pepper, bell - Bacterial spot | 1.0 | 0.993 | 0.997 | 150.0 |
| Pepper, bell - healthy | 1.0 | 1.0 | 1.0 | 222.0 |
| Potato - Early blight | 1.0 | 1.0 | 1.0 | 150.0 |
| Potato - Late blight | 1.0 | 1.0 | 1.0 | 150.0 |
| Potato - healthy | 1.0 | 1.0 | 1.0 | 23.0 |
| Raspberry - healthy | 1.0 | 1.0 | 1.0 | 56.0 |
| Soybean - healthy | 0.999 | 1.0 | 0.999 | 764.0 |
| Squash - Powdery mildew | 1.0 | 1.0 | 1.0 | 275.0 |
| Strawberry - Leaf scorch | 1.0 | 1.0 | 1.0 | 167.0 |
| Strawberry - healthy | 1.0 | 1.0 | 1.0 | 69.0 |
| Tomato - Bacterial spot | 0.984 | 0.984 | 0.984 | 319.0 |
| Tomato - Early blight | 0.987 | 1.0 | 0.993 | 150.0 |
| Tomato - Late blight | 1.0 | 0.989 | 0.995 | 285.0 |
| Tomato - Leaf Mold | 1.0 | 1.0 | 1.0 | 143.0 |
| Tomato - Septoria leaf spot | 0.996 | 0.996 | 0.996 | 266.0 |
| Tomato - Spider mites Two-spotted spider mite | 0.996 | 1.0 | 0.998 | 252.0 |
| Tomato - Target Spot | 0.977 | 0.995 | 0.986 | 211.0 |
| Tomato - Tomato Yellow Leaf Curl Virus | 1.0 | 0.996 | 0.998 | 804.0 |
| Tomato - Tomato mosaic virus | 1.0 | 1.0 | 1.0 | 56.0 |
| Tomato - healthy | 1.0 | 1.0 | 1.0 | 238.0 |

Most frequent confusions:
| true | predicted | count |
|---|---|---|
| Corn (maize) - Northern Leaf Blight | Corn (maize) - Cercospora leaf spot Gray leaf spot | 7 |
| Tomato - Bacterial spot | Tomato - Target Spot | 5 |
| Tomato - Late blight | Tomato - Bacterial spot | 2 |
| Tomato - Tomato Yellow Leaf Curl Virus | Tomato - Bacterial spot | 2 |
| Apple - healthy | Blueberry - healthy | 1 |
| Cherry (including sour) - healthy | Soybean - healthy | 1 |
| Corn (maize) - Cercospora leaf spot Gray leaf spot | Corn (maize) - Northern Leaf Blight | 1 |
| Pepper, bell - Bacterial spot | Tomato - Bacterial spot | 1 |
| Tomato - Late blight | Tomato - Early blight | 1 |
| Tomato - Septoria leaf spot | Tomato - Early blight | 1 |
| Tomato - Target Spot | Tomato - Septoria leaf spot | 1 |
| Tomato - Tomato Yellow Leaf Curl Virus | Tomato - Spider mites Two-spotted spider mite | 1 |

## 5. Confidence & low-confidence threshold
Confidence = max softmax probability after temperature scaling (fitted on the validation set).
Threshold **0.70** was chosen on validation images (clean + 14 simulated difficult conditions) as the lowest value
(>= 0.7) where accepted predictions are >= 98% accurate.
Clean test at this threshold: coverage 0.998, accuracy of accepted 0.998.
Below the threshold the API returns `LOW_CONFIDENCE` and **no prediction**, only candidates. Plot: `threshold_analysis.png`.

## 6. Difficult conditions (stress tests on test subset)
Synthetic lighting / colour cast / blur / noise / low resolution / compression / close-up / background replacement. Background swaps are *procedural
approximations* and the model saw similar augmentation in training, so treat them as optimistic.
| condition | accuracy | mean_confidence | coverage | selective_accuracy | wrong_but_accepted |
|---|---|---|---|---|---|
| clean | 0.999 | 0.998 | 0.999 | 0.999 | 0.001 |
| dark | 0.999 | 0.998 | 0.999 | 0.999 | 0.001 |
| overexposed | 0.988 | 0.994 | 0.992 | 0.993 | 0.007 |
| low_contrast | 0.996 | 0.998 | 0.999 | 0.996 | 0.004 |
| warm_cast | 0.958 | 0.98 | 0.98 | 0.965 | 0.034 |
| cool_cast | 0.946 | 0.974 | 0.962 | 0.959 | 0.039 |
| blur | 0.955 | 0.976 | 0.967 | 0.97 | 0.029 |
| noise | 0.924 | 0.974 | 0.97 | 0.94 | 0.058 |
| low_res | 0.946 | 0.978 | 0.975 | 0.957 | 0.042 |
| jpeg_q10 | 0.986 | 0.987 | 0.98 | 0.991 | 0.009 |
| close_up | 0.971 | 0.988 | 0.991 | 0.972 | 0.028 |
| bg_soil | 1.0 | 0.998 | 0.999 | 1.0 | 0.0 |
| bg_grass | 1.0 | 0.998 | 0.999 | 1.0 | 0.0 |
| bg_gray | 1.0 | 0.998 | 0.997 | 1.0 | 0.0 |
| field_combo | 0.982 | 0.988 | 0.983 | 0.989 | 0.011 |

## 7. Real-world field test
No images supplied in data/field_test/. **Strongly recommended before production.**

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
