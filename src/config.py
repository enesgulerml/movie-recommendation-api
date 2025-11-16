# src/config.py

from pathlib import Path

# === 1. File Paths (Dynamic and Absolute) ===
SRC_ROOT = Path(__file__).parent
PROJECT_ROOT = SRC_ROOT.parent

# --- Raw Data Paths ---
RATINGS_DATA_FILE = "ratings.dat"
MOVIES_DATA_FILE = "movies.dat"

RATINGS_DATA_PATH = PROJECT_ROOT / "data" / "raw" / RATINGS_DATA_FILE
MOVIES_DATA_PATH = PROJECT_ROOT / "data" / "raw" / MOVIES_DATA_FILE

# --- Model Output Path ---
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "recsys_svd_model.pkl"

# --- "Polish" Data Path for API (v3.0) ---
MOVIES_CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "movies_cleaned.pkl"


# === 2. Data Loading Settings (CHAOS CLEANING) ===
DATA_SEPARATOR = "::"
DATA_ENGINE = "python"
MOVIES_ENCODING = "latin-1"

RATINGS_COLS = ["UserID", "MovieID", "Rating", "Timestamp"]
MOVIES_COLS = ["MovieID", "Title", "Genres"]


# === 3. v1.0 Model and Pipeline Settings ===

TARGET_VARIABLE = "Rating"

# Train-test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

SVD_PARAMS = {
    'n_factors': 100,
    'n_epochs': 20,
    'lr_all': 0.005,
    'reg_all': 0.02,
    'random_state': RANDOM_STATE
}


# === 4. MLFlow (v2.0) and API (v3.0) Settings ===
MLFLOW_EXPERIMENT_NAME = "Movie Recommendation System (SVD)"

# How many recommendations should the API (v3.0) return for a user?
TOP_K_RECOMMENDATIONS = 10