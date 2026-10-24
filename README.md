# Crop Disease AI (`cropdx`)

**Crop Disease AI** is an end-to-end computer vision and deep learning framework designed to detect, diagnose, and analyze plant leaf diseases. Built to handle the domain shift between lab-captured datasets (like PlantVillage) and real-world agricultural conditions, the platform includes tools for data preprocessing, augmentation, background perturbation, model training, explainability (Grad-CAM), REST API deployment, and enterprise integration.

---

## 🏗️ System Architecture & Workflow

The system provides a complete pipeline from raw imagery to real-time production inference:

```
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│  Raw Datasets   │ ──> │ Data Preparation     │ ──> │ Model Training   │
│ (PlantVillage/  │     │ & Perturbation       │     │ (ResNet, Efficient│
│ Field Images)   │     │ (Dedupe/Backgrounds) │     │  Net, MobileNet) │
└─────────────────┘     └──────────────────────┘     └─────────┬────────┘
                                                               │
                                                               ▼
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│ Enterprise Apps │ <── │  FastAPI Server      │ <── │ Evaluation &     │
│ (Spring Boot)   │     │  (/diagnose, Grad-CAM│     │  Export (ONNX/   │
└─────────────────┘     └──────────────────────┘     │  TorchScript)    │
                                                     └──────────────────┘
```

---

## ✨ Core Features

* **Data Preparation & Robustness Enhancement (`cropdx/prepare_data.py`, `perturb.py`, `dedupe.py`)**:
  * Automated deduplication of leaf images to prevent data leakage between split sets.
  * Background removal and synthetic background insertion/perturbation to prevent model overfitting on lab-environment backgrounds.
  * Ingestion and formatting for field-captured datasets.

* **Flexible Model Training (`cropdx/train.py`, `models.py`)**:
  * Deep learning model zoo for plant pathology classification (e.g., ResNet, MobileNet, EfficientNet variants).
  * Hyperparameter management via `config.yaml`.

* **Explainability & Diagnostics (`cropdx/gradcam.py`, `report.py`, `diagnose.py`)**:
  * **Grad-CAM Integration**: Generates visual heatmaps to highlight region-of-interest indicators for disease diagnosis.
  * PDF and visual diagnostic report generation summarizing diagnostic confidence, disease descriptions, and recommended actions.

* **Model Evaluation & Export (`cropdx/evaluate.py`, `export.py`, `metrics.py`)**:
  * Model evaluation across diverse subsets (lab vs. field validation sets).
  * Export options for deployment in constrained environments (TorchScript, ONNX, and quantized formats).

* **Production REST API (`api/main.py`)**:
  * High-performance FastAPI backend serving model predictions and Grad-CAM visualizations over HTTP endpoints.
  * Structured request/response validation via Pydantic schemas.

* **Enterprise Integration (`integration/spring-boot-sample/`)**:
  * Java / Spring Boot client implementation (`CropDiseaseClient`, `DiagnosisController`) demonstrating how to integrate the Python API into enterprise microservice architectures.

---

## 📁 Repository Structure

```
crop-disease-ai/
├── api/                        # FastAPI REST service
│   ├── main.py                 # API endpoints
│   └── schemas.py              # Request/Response validation schemas
├── cropdx/                     # Core Python engine & library
│   ├── check_model.py          # Model verification utilities
│   ├── config.py               # Config loader
│   ├── constants.py            # Global definitions and paths
│   ├── data.py                 # PyTorch DataLoaders & Datasets
│   ├── dedupe.py               # Image deduplication
│   ├── diagnose.py             # Inference & diagnosis workflow
│   ├── evaluate.py             # Performance metrics & evaluation
│   ├── export.py               # TorchScript / ONNX exporters
│   ├── gradcam.py              # Grad-CAM heatmap generation
│   ├── import_field_dataset.py # Field image ingestion tools
│   ├── labels.py               # Crop disease label mappings
│   ├── metrics.py              # Custom evaluation metrics
│   ├── models.py               # Neural network architectures
│   ├── perturb.py              # Background perturbation pipeline
│   ├── prepare_data.py         # Pipeline data prep script
│   ├── preprocess.py           # Image transformations
│   ├── report.py               # PDF/Image report generator
│   └── train.py                # Training loop
├── data/                       # Data storage directory
│   ├── backgrounds/            # Synthetic background images
│   ├── field_test/             # Real-world test dataset
│   ├── field_train/            # Real-world training dataset
│   └── raw/                    # Raw input directory (PlantVillage, etc.)
├── docs/                       # API specs and architecture guides
│   ├── API_CONTRACT.md         # Open API specifications
│   └── HANDOVER.md             # System handover guide
├── integration/                # Sample client integrations
│   └── spring-boot-sample/     # Spring Boot Java integration code
├── tests/                      # Unit and integration test suite
├── config.yaml                 # Central configuration settings
├── docker-compose.yml          # Container orchestration configuration
├── Dockerfile                  # API service container specification
├── run_pipeline.py             # Complete CLI pipeline entrypoint
├── requirements.txt            # Core engine dependencies
└── requirements-api.txt        # API dependencies
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* Docker & Docker Compose (optional for containerized deployment)

### Installation

Clone the repository and install dependencies:

```bash
# Core engine dependencies
pip install -r requirements.txt

# API dependencies (if running the server)
pip install -r requirements-api.txt
```

## 🌐 Serving the API

### Direct Python Execution

Launch the FastAPI server using Uvicorn:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Access the interactive API documentation at: `http://localhost:8000/docs`

### Docker Deployment

To build and launch the API container:

```bash
docker-compose up --build
```

---

## 🔌 API Endpoints Summary

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | `GET` | Health check endpoint |
| `/diagnose` | `POST` | Upload a crop image to receive class predictions and confidence scores |
| `/diagnose/explain` | `POST` | Get prediction along with a Grad-CAM heatmap visualization |

---

## 🧪 Testing

Run tests across core modules and field tools using `pytest`:

```bash
pytest tests/
```
