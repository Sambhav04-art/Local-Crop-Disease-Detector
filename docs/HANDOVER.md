# What to give the backend developer + what to agree with her

## A. Deliver to her (pick the option you agree on)
**Option 1 (recommended): run the service as a container/process, she only calls HTTP.**
1. `docs/API_CONTRACT.md` (the contract) and the live `/docs` + `/openapi.json` (she can generate a Java client from it, optional).
2. The Docker image or the folder: `api/`, `cropdx/{__init__,constants,preprocess,labels}.py`, `artifacts/` (model.onnx, labels.json, model_meta.json), `requirements-api.txt`, `Dockerfile`, `docker-compose.yml`.
3. `reports/MODEL_CARD.md` + `reports/EVALUATION_REPORT.md` (limitations, supported classes, threshold).
4. `integration/spring-boot-sample/` (ready RestClient + DTO records + Mongo document + controller).
5. **Before the real model exists:** `python tests/make_dummy_model.py` then `MODEL_DIR=artifacts_dummy uvicorn api.main:app --port 8000`
   gives her the exact same API with random predictions (`modelVersion` = `0.0.0-DUMMY`), so she can build and test in parallel.
6. Three values: service base URL per environment, the `API_KEY`, and the expected max image size.

**Option 2 (alternative): she runs the model inside Java** with ONNX Runtime Java (`com.microsoft.onnxruntime`) using `artifacts/model.onnx`.
Then she must re-implement preprocessing exactly (see `cropdx/preprocess.py`: EXIF rotate, RGB, resize 224x224 bilinear, /255, ImageNet mean/std, CHW) and
the temperature + softmax + threshold in `model_meta.json`. More work and easy to get subtly wrong; use only if you cannot run a Python service.

## B. Things to agree in a 30-minute meeting (write the decisions into the contract)
* Who validates uploads (type/size) first - suggestion: Spring, then forwards bytes.
* Where images are stored (GridFS / S3) and retention; whether users consent to their photos being used for retraining.
* What Spring stores: whole `PredictionResponse` + `modelVersion` + image reference (see `DiagnosisRecord`).
* How low-confidence is shown to users; who owns translation of `crop`/`condition` (key on `classId`).
* Timeouts/retries, deployment (same Docker network, port 8000 not public), monitoring of `/health`, logging of `requestId`.
* Model update policy: `modelVersion` changes => she re-runs her contract tests (`tests/` + `/docs`); breaking changes only via `/v2`.
* Optional feedback endpoint on her side ("was this correct?") - this is how you later get real field data to improve the model.

## C. What you (ML side) must be able to answer
Supported classes (GET /v1/classes), the threshold and why, expected accuracy and its limits (MODEL_CARD), CPU/RAM needs (~1 CPU, < 1 GB), cold-start time (a few seconds).
