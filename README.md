# 🎬 End-to-End Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)
![Testing](https://img.shields.io/badge/Tests-Pytest-brightgreen)

## 📖 Overview
This repository hosts a production-ready **MLOps pipeline** for movie recommendations based on the MovieLens 1M dataset. It demonstrates a complete lifecycle from raw data processing to model serving via REST API and an interactive dashboard.

**Key Features:**
* **Inference Engine:** SVD Model served via FastAPI (High Performance).
* **Frontend:** Decoupled Streamlit dashboard for user interaction.
* **Reproducibility:** Fully Dockerized environment using explicit volume mounts.
* **Quality Assurance:** Automated E2E and Unit tests via Pytest.
* **Experiment Tracking:** MLflow integration for model metrics.

---

## 📂 Project Structure

```text
movie-recommendation-api/
│
├── app/                  # API Service (FastAPI)
│   ├── main.py           # Application Entry Point
│   └── schema.py         # Pydantic Data Contracts
│
├── dashboard/            # Frontend (Streamlit)
│   └── app.py            # UI Logic
│
├── src/                  # ML Pipeline (Training & Processing)
│   ├── config.py         # Hyperparameters & Paths
│   ├── train.py          # Training Script (SVD + GridSearchCV)
│   └── data_processing.py
│
├── tests/                # Automated Test Suite
├── requirements.txt      # Production Dependencies
└── Dockerfile            # Container Configuration
```

## 🛠️ Installation & Setup
Prerequisites
* Python 3.10+
* Docker (Optional but recommended)
* [MovieLens 1M Dataset](https://www.kaggle.com/code/kushtrivedi1/movie-recommendation-on-movielens1m) (Place in data/raw/)

### 1. Environment Setup
We recommend using a fresh virtual environment to avoid dependency conflicts.

```bash
# Clone the repository
git clone https://github.com/enesgulerml/movie-recommendation-api.git
cd movie-recommendation-api

# Create Environment
conda create -n movie-rec-sys python=3.10
conda activate movie-rec-sys

# Install Dependencies
pip install -r requirements.txt
pip install -e .
```

## 🚀 How to Run
Since the trained model files are not included in the repository (due to size limits), you must train the model locally first.

### Option A: Train the Model (Required First Step)
This pipeline processes the raw data, trains the SVD model, and saves the artifacts to the models/ directory.

```bash
# Run the training pipeline
python -m src.train
```
✅ Success: Check that models/recsys_svd_model.pkl has been created.

### Option B: Run API with Docker
Once the model is trained, use Docker to serve the API. We mount your local models/ folder so the container can access the model you just created.

1. Build the Image:
```bash
docker build -t recsys-api:latest .
```

2. Run the Container:
```bash
docker run -d --rm -p 8000:80 \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/data:/app/data" \
  recsys-api:latest
```
👉 Access API Docs: http://localhost:8000/docs

### Option C: User Dashboard (Frontend)
To launch the interactive frontend (ensure API is running first):
```
streamlit run dashboard/app.py
```
👉 Access Dashboard: http://localhost:8501

## 🧪 Testing
The project includes a robust test suite to ensure data integrity and API availability.
```bash
# Run all tests
pytest

# Run only fast tests (skip integration)
pytest -m "not slow"
```