# API Contract - Crop Disease Prediction Service (v1)

Base URL (internal): `http://<host>:8000`  |  Interactive docs: `/docs`  |  Machine-readable: `/openapi.json`
JSON fields are **camelCase**. Unknown fields must be ignored by the client (we may add fields without a version bump).
Breaking changes => new path prefix `/v2`. Every response carries `X-Request-Id` (send your own in the same header to correlate logs).

**Auth:** if the service runs with `API_KEY`, send `X-API-Key: <key>` on every call except `/health`.
**Network:** the service must NOT be exposed to the internet. Only the Spring Boot backend calls it; Spring handles user auth.

## POST /v1/predict
`multipart/form-data`, part name **`file`** (phone EXIF rotation is applied; non-square photos are centre-cropped to a square, so the leaf should be near the middle) (JPEG/PNG/WebP, max 8 MB, min 32 px). Optional query `topK` (1-5, default 3).

```
curl -X POST "http://localhost:8000/v1/predict?topK=3" -H "X-API-Key: change-me" -F "file=@leaf.jpg"
```

### 200 OK
```json
{
  "requestId": "6f1c...",
  "modelVersion": "1.0.0",
  "status": "CONFIDENT",
  "prediction": {"classId": "Tomato___Early_blight", "crop": "Tomato", "condition": "Early blight",
                 "displayName": "Tomato - Early blight", "healthy": false, "confidence": 0.9712},
  "candidates": [ {"classId": "...", "crop": "...", "condition": "...", "displayName": "...", "healthy": false, "confidence": 0.9712}, ... ],
  "confidenceThreshold": 0.80,
  "qualityWarnings": ["IMAGE_BLURRY"],
  "message": "The model indicates Tomato: Early blight (97% confidence). This is an AI-assisted indication, not a confirmed diagnosis.",
  "disclaimer": "AI-assisted indication only. This is not a guaranteed agricultural diagnosis. ...",
  "inferenceMs": 31.4
}
```
| field | meaning |
|---|---|
| `status` | `CONFIDENT`: top confidence >= `confidenceThreshold`. `LOW_CONFIDENCE`: below it. **Branch on this field, never on your own threshold.** |
| `prediction` | top-1 class. **`null` when `LOW_CONFIDENCE`** (do not show a diagnosis then). |
| `candidates` | top-k, sorted desc. Safe to show as "possible matches" even when low confidence. |
| `confidence` | calibrated probability 0-1 (4 decimals). It is an indication of model certainty, **not** a probability that the farmer's plant truly has the disease. |
| `qualityWarnings` | any of `LOW_RESOLUTION`, `IMAGE_TOO_DARK`, `IMAGE_TOO_BRIGHT`, `IMAGE_BLURRY` (heuristic hints for "retake photo"). |
| `disclaimer` | **must be displayed to the end user** with every result. |
| `modelVersion` | store with every saved result (audit, later re-analysis). |
| `classId` | stable identifier (e.g. use as key in your DB / i18n tables). `crop`, `condition`, `displayName` are English display texts. |

### Errors (all use this envelope)
```json
{"requestId": "6f1c...", "error": {"code": "INVALID_IMAGE", "message": "File is not a readable JPEG/PNG/WebP image."}}
```
| HTTP | code | when | backend should |
|---|---|---|---|
| 400 | `INVALID_IMAGE` / `EMPTY_FILE` | unreadable, tiny or empty image | tell user to upload another photo (return 422 to your client) |
| 401 | `UNAUTHORIZED` | bad/missing API key | alert ops, config error |
| 413 | `FILE_TOO_LARGE` | > 8 MB | compress/reject before forwarding |
| 422 | `VALIDATION_ERROR` | wrong field name / topK out of range | fix the call (bug) |
| 503 | `MODEL_UNAVAILABLE` | model not loaded | retry once, then show "service temporarily unavailable" |
| 500 | `INTERNAL_ERROR` | unexpected | retry once, log `requestId` |

Recommended client timeouts: connect 3 s, read 15 s (typical inference < 100 ms on CPU). Retry only on 503/connection errors, max 1 retry. The call is read-only/idempotent.

## GET /v1/classes
`[ {"classId","crop","condition","displayName","healthy"} ... ]` - the complete supported list (drive your crop/disease dropdowns from this).

## GET /v1/model-info
`{"modelVersion","architecture","imageSize","numClasses","confidenceThreshold","testAccuracy","testMacroF1","trainedDataset","exportedAt","disclaimer","supportedCrops":[...]}`

## GET /health  (no auth)
`200 {"status":"UP","modelLoaded":true}` or `503 {"status":"DOWN","modelLoaded":false}` - use for readiness probes.

## Mandatory UX/legal rules for the product using this API
1. Always show `disclaimer`. Never word a result as a diagnosis ("Your plant HAS..."); use "likely / possible".
2. For `LOW_CONFIDENCE` show "couldn't identify - retake photo" (+ optional candidates), never a single confident-looking answer.
3. Show `qualityWarnings` as photo tips. 4. Never give treatment/dosage advice based solely on this output.
