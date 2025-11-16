# src/train.py

import pandas as pd
import sys
import datetime
import mlflow
import mlflow.sklearn
import warnings
from warnings import filterwarnings
filterwarnings("ignore")
from joblib import dump

# --- Surprise Library Imports ---
# This is the "Google-level" library for Recommender Systems
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy  # This is for calculating RMSE

# --- Our Project Imports ---
from src.config import (
    MODEL_OUTPUT_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    TARGET_VARIABLE,  # "Rating"
    MLFLOW_EXPERIMENT_NAME,
    SVD_PARAMS  # The hyperparameters for our algorithm
)
from src.data_processing import load_ratings_data, load_and_save_movies_data


def run_training():
    """
    v2.0 - Main training orchestrator for the Recommendation System.

    1. (v2.0) Starts MLFlow experiment.
    2. (v1.0) Loads ratings data.
    3. (v1.0) Converts data to Surprise 'Dataset' format.
    4. (v1.0) Splits into train/test sets.
    5. (v1.0) Defines and trains the SVD algorithm.
    6. (v1.0) Calculates RMSE metric.
    7. (v2.0) Logs params, metrics, and model to MLFlow.
    8. (v1.0) Saves the final model to 'models/'.
    9. (v3.0) Prepares the 'movies.dat' file for the API.
    """
    print("===== Starting Training Process (v2.0 - MLFlow) =====")

    # === Start the MLFlow Experiment ===
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"run_recsys_{current_time}"

    with mlflow.start_run(run_name=run_name):
        # --- MLFlow Logging Step 1: Parameters ---
        print("Logging parameters to MLFlow...")
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_params(SVD_PARAMS)  # Log all SVD HParams

        # --- Step 1: Load Ratings Data ---
        ratings_df = load_ratings_data()

        # --- Step 2: Convert to Surprise Dataset ---
        # The 'surprise' library needs a 'Reader' to parse the DataFrame
        # We tell it our ratings are on a scale of 1 to 5.
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            ratings_df[["UserID", "MovieID", "Rating"]],
            reader
        )
        print("Pandas DataFrame converted to Surprise Dataset.")

        # --- Step 3: Train/Test Split ---
        trainset, testset = train_test_split(
            data,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        # --- Step 4: Define and Train SVD Algorithm ---
        print("Defining SVD algorithm with config parameters...")
        # We use the "Google-level" SVD algorithm
        # We unpack our hyperparameters from config.py
        algo = SVD(**SVD_PARAMS)

        print("Training (fitting) SVD model...")
        # Train the algorithm on the trainset
        algo.fit(trainset)
        print("Model training complete.")

        # --- Step 5: Evaluate Model (Calculate RMSE) ---
        print("Generating predictions on the testset...")
        predictions = algo.test(testset)

        # 'accuracy' in Surprise library means 'RMSE' (Root Mean Squared Error)
        # For ratings, lower RMSE is better (less error).
        rmse = accuracy.rmse(predictions)
        print(f"Model Test RMSE: {rmse:.4f}")

        # --- MLFlow Logging Step 2: Metrics ---
        print("Logging metrics to MLFlow...")
        mlflow.log_metric("rmse", rmse)

        # --- Step 6: Save Model (Local + MLFlow) ---

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

        # --- Step 7: (v3.0 Cila) Prepare Movies data for API ---
        # This is a one-time step to create the 'movies_cleaned.pkl'
        # file that our v3.0 API will use.
        print("Preparing clean movie data (movies_cleaned.pkl) for the API...")
        load_and_save_movies_data()

        print("===== Training Process Completed (MLFlow) =====")


if __name__ == "__main__":
    run_training()