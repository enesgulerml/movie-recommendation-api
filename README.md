# End-to-End Movie Recommendation System (v5.3)

This project builds a "Google-level," production-ready MLOps pipeline for a **Movie Recommendation System** based on the MovieLens 1M dataset.

This system demonstrates a full end-to-end lifecycle, converting raw `.dat` files into a "polished" (human-readable) API service.

* **v1.0: Feature Engineering & Model (SVD)**
* **v2.0: Experiment Tracking (MLFlow + GridSearchCV)**
* **v3.1: API Serving (FastAPI + Docker)**
* **v4.0: Interactive Dashboard (Streamlit)**
* **v5.3: Full Automated Testing (Pytest)**

---

## 🚀 Project Structure

```
movie-recommendation-api/
│
├── app/                  <- (v3.0) API service code (FastAPI)
│   ├── main.py           <- (API "Motor")
│   └── schema.py         <- (Pydantic "Contract")
│
├── dashboard/            <- (v4.0) Streamlit dashboard code
│   └── app.py            <- (Dashboard "Direksiyon")
│
├── data/
│   ├── raw/
│   │   ├── ratings.dat     <- (Raw 1M ratings, *not* tracked by Git)
│   │   └── movies.dat      <- (Raw movie titles, *not* tracked by Git)
│   └── processed/
│       └── movies_cleaned.pkl <- (v3.0 "Polish" for API, *not* tracked by Git)
│
├── models/
│   └── recsys_svd_model.pkl <- (Trained SVD model, *not* tracked by Git)
│
├── mlruns/                 <- (v2.0) MLFlow experiment logs, *not* tracked by Git)
│
├── src/                  <- (v1.0 & v2.0) All training & data loading code
│   ├── config.py             <- (All settings, paths, and HParams)
│   ├── data_processing.py    <- (Loads raw .dat files)
│   └── train.py              <- (Main training script - SVD + GridSearchCV)
│
├── test/                 <- (v5.3) Automated "Safety Net"
│   ├── test_api_e2e.py       <- (E2E API Test)
│   ├── test_pipeline.py      <- (Unit Test)
│   └── test_train_integration.py <- (Integration Test)
│
├── .dockerignore             <- (Tells Docker which files to ignore)
├── .gitignore                <- (Tells Git which files to ignore)
├── requirements.txt          <- (v5 Strategy: The *only* source of truth for dependencies)
├── setup.py                  <- (Makes 'src' an installable package)
├── pytest.ini                <- (Configures pytest markers)
└── README.md                 <- (This file - The project user manual)
```

---

## 🛠️ Installation & Setup (v5 Strategy)

Follow these steps to set up the project environment on your local machine.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/enesgulerml/movie-recommendation-api.git
    cd movie-recommendation-api
    ```

2.  **Download the Data (MovieLens 1M):**
    * Download `ratings.dat`, `users.dat`, and `movies.dat` (e.g., from Kaggle).
    * Place these files inside the `data/raw/` directory (you may need to create this folder).

3.  **Create Conda Environment (Base only):**
    We only use Conda to manage Python itself, not packages (to avoid Conda's dependency "kaos").
    ```bash
    conda create -n movie-recommendation-api python=3.10
    conda activate movie-recommendation-api
    ```

4.  **Install Dependencies (v5 Strategy - Pip):**
    This command reads the `requirements.txt` file (the *single source of truth*).
    ```bash
    pip install -r requirements.txt
    ```

5.  **Install the Project Package:**
    This "Google-level" step makes your `src` code importable.
    ```bash
    pip install -e .
    ```
    Note: If you encounter environment-specific errors (like conda not found, docker build memory (cannot allocate memory), or pip install C++ compiler (gcc failed/Visual C++) errors), please check our TROUBLESHOOTING.md guide.

---

## 🧪 v5.3: Running Automated Tests (Pytest)

This project includes a "Google-level" safety net of automated tests (`test/` directory).

### 1. Run All Tests (Fast & Slow)
This will run all tests (Unit, Integration, and E2E).
*(Note: This requires Docker to be running and the `recsys-api:v3` image to be built.)*

```bash
python -m pytest
```
*Expected Output: `== 5 passed ==`*

### 2. Run Only Fast Unit Tests
This skips any test marked as `@pytest.mark.slow`.

```bash
python -m pytest -m "not slow"
```
*Expected Output: `== 2 passed, 3 deselected ==`*

---

## ⚡ How to Use (v1.0 - v4.0)

Once tested, you can use the project's features.

### 1. v2.1: Train Model & Track (MLFlow)

This is the main "orchestrator" script. It runs the entire pipeline (Data Processing, GridSearchCV, Training, Tracking).

```bash
python -m src.train
```

To view the results and compare different runs, launch the MLFlow dashboard:
```bash
mlflow ui
```

### 2. v3.1: Serve Model (FastAPI + Docker)

This runs the v3.1 API server in a Docker container (using our "kaos-free" v5 strategy `Dockerfile`).

#### 1. Build the v3.1 API Image
(If you encounter `cannot allocate memory` or `gcc failed` errors, this project has been refactored to use `pip` and `python:3.10-slim` to solve this "kaos".)
```bash
docker build -t recsys-api:v3 .
```

#### 2. Run the API Server (Docker)
This command runs the API in "detached" mode (`-d`), maps your local port `8000` (`-p 8000:80`), and crucially, mounts all 3 required "data" directories (`-v`).

```bash
docker run -d --rm -p 8000:80 \
  -v ${pwd}/models:/app/models \
  -v ${pwd}/data/processed:/app/data/processed \
  -v ${pwd}/data/raw:/app/data/raw \
  recsys-api:v3
```

#### 3. Test the API
Once the container is running (check with `docker ps`), go to your browser:

* **API Docs (Swagger):** `http://localhost:8000/docs`
* **Test Endpoint:** Try `GET /recommend/{user_id}` with `user_id = 5`

---

### 3. v4.0: View Dashboard (Streamlit)

This dashboard (`dashboard/app.py`) is a **decoupled frontend** that consumes the v3.1 API Container.

This requires **two separate terminals** running simultaneously:

**➡️ Terminal 1: Run the API Server (v3.1)**
(If not already running) Start the FastAPI Docker container:
```bash
docker run -d --rm -p 8000:80 \
  -v ${pwd}/models:/app/models \
  -v ${pwd}/data/processed:/app/data/processed \
  -v ${pwd}/data/raw:/app/data/raw \
  recsys-api:v3
```

**➡️ Terminal 2: Run the Streamlit App (v4.0)**
Activate the conda environment and run the Streamlit app:
```bash
conda activate movie-recommendation-api
python -m streamlit run dashboard/app.py
```
Your browser will open `http://localhost:8501`, where you can interact with the live system.