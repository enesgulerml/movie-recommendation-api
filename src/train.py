# src/train.py

import pandas as pd
import sys
import datetime
import mlflow
import mlflow.sklearn
from joblib import dump

# --- Surprise Library Imports ---
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import GridSearchCV
from surprise import accuracy

# --- Our Project Imports ---
from src.config import (
    MODEL_OUTPUT_PATH,
    RANDOM_STATE,
    MLFLOW_EXPERIMENT_NAME,
    GRID_SEARCH_CV_PARAMS,  # v2.2: The HParam search space
    TARGET_VARIABLE  # "Rating"
)
from src.data_processing import load_ratings_data, load_and_save_movies_data


def run_training():
    """
    v2.2 - Main training orchestrator with Hyperparameter Tuning.

    1. (v2.0) Starts MLFlow experiment.
    2. (v1.0) Loads ratings data.
    3. (v1.0) Converts data to Surprise 'Dataset' format.
    4. (v2.2) Defines and runs GridSearchCV to find the best SVD hyperparameters.
    5. (v2.2) Re-trains the single best algorithm on the *full* dataset.
    6. (v2.0) Logs best params and metrics (RMSE) to MLFlow.
    7. (v1.0) Saves the final, re-trained model to 'models/'.
    8. (v3.0) Prepares the 'movies.dat' file (as .pkl) for the API.
    """
    print("===== Starting Training Process (v2.2 - GridSearchCV) =====")

    # === Start the MLFlow Experiment ===
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"run_recsys_gridsearch_{current_time}"

    with mlflow.start_run(run_name=run_name):
        # --- MLFlow Logging Step 1: Parameters (The Search Space) ---
        print("Logging search space parameters to MLFlow...")
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("cv_folds", GRID_SEARCH_CV_PARAMS['cv'])

        # --- Step 1: Load Ratings Data ---
        ratings_df = load_ratings_data()

        # --- Step 2: Convert to Surprise Dataset ---
        # The 'surprise' library needs a 'Reader' to parse the DataFrame
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            ratings_df[["UserID", "MovieID", TARGET_VARIABLE]],
            reader
        )
        print("Pandas DataFrame converted to Surprise Dataset.")

        # --- Step 3 (v2.2): Define and Run GridSearchCV ---
        print("Defining GridSearchCV with config parameters...")
        gs = GridSearchCV(
            SVD,  # The algorithm class to test
            **GRID_SEARCH_CV_PARAMS  # Unpack 'param_grid', 'measures', 'cv', etc.
        )

        print("Running GridSearchCV (this may take several minutes)...")
        # Fit on the full dataset (it handles Cross-Validation internally)
        gs.fit(data)

        print("GridSearchCV complete.")

        # --- Step 4 (v2.2): Get Best Model and Re-train ---

        # Get the best RMSE score from the grid search results
        best_rmse_cv = gs.best_score['rmse']
        print(f"Best Model (from GridSearchCV CV) RMSE: {best_rmse_cv:.4f}")

        # Get the best parameters found
        best_params = gs.best_params['rmse']
        print(f"Best parameters found: {best_params}")

        # Best Practice: Re-train the best algorithm on the FULL dataset
        # before saving it for production.
        print("Re-training the best model on the full dataset...")

        # Add the 'random_state' from config to the best params
        best_params['random_state'] = RANDOM_STATE

        algo = SVD(**best_params)
        full_trainset = data.build_full_trainset()
        algo.fit(full_trainset)

        print("Best model re-training complete.")

        # --- MLFlow Logging Step 2: Metrics & Best Params ---
        print("Logging best metric and params to MLFlow...")
        mlflow.log_metric("best_rmse_cv", best_rmse_cv)  # Log the CV score
        mlflow.log_params(best_params)  # Log the *winning* params

        # --- Step 5: Save Model (Local + MLFlow) ---

        # Local Save (for our API)
        MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        dump(algo, MODEL_OUTPUT_PATH)
        print(f"Trained model saved to: {MODEL_OUTPUT_PATH}")

        # MLFlow Save (for tracking)
        print("Logging model (artifact) to MLFlow...")
        mlflow.sklearn.log_model(
            sk_model=algo,
            name="recsys_svd_model",
            input_example=ratings_df.head(10)
        )

        # --- Step 6: (v3.0 Polish) Prepare Movies data for API ---
        print("Preparing clean movie data (movies_cleaned.pkl) for the API...")
        load_and_save_movies_data()

        print("===== Training Process Completed (MLFlow + GridSearchCV) =====")


if __name__ == "__main__":
    run_training()