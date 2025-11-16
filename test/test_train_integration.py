# test/test_train_integration.py

import pytest
import pandas as pd
from pathlib import Path

# --- Import the *correct* components ---
from src.data_processing import load_ratings_data
from src.config import (
    TEST_SIZE,
    RANDOM_STATE,
    TARGET_VARIABLE
)

# --- Surprise Library Imports (REQUIRED FOR THIS TEST) ---
from surprise import Dataset, Reader, SVD
from surprise import accuracy  # This calculates RMSE
from surprise.model_selection import train_test_split  # This is from 'surprise'

# === v5.2.2 "Kaos" Fix ===
# A "Google-level" test should not depend on experimental configs.
# We define static, fast parameters just for this test.
TEST_SVD_PARAMS = {
    'n_factors': 50,  # Simple params just for testing
    'n_epochs': 5,  # Run fast (5 epochs, not 20)
    'lr_all': 0.005,
    'reg_all': 0.02,
    'random_state': RANDOM_STATE
}


@pytest.mark.slow
def test_full_training_integration_pipeline():
    """
    Test v5.2.2 (Integration Test):
    Validates the v1.0 (Data Processing) and a v1.0 (SVD Training) pipeline.
    """

    # --- 1. Arrange ---
    try:
        ratings_df = load_ratings_data()
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            ratings_df[["UserID", "MovieID", "Rating"]],
            reader
        )
    except FileNotFoundError:
        pytest.fail(
            "Integration Test Failed: 'data/raw/ratings.dat' not found."
        )
    except Exception as e:
        pytest.fail(f"Integration Test Failed: Data loading raised: {e}")

    # --- 2. Act ---
    # Use 'surprise's' own train_test_split
    trainset, testset = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Use the static 'TEST_SVD_PARAMS'
    algo = SVD(**TEST_SVD_PARAMS)

    algo.fit(trainset)
    predictions = algo.test(testset)

    # Use 'surprise's' own 'accuracy.rmse'
    rmse = accuracy.rmse(predictions, verbose=False)  # verbose=False to quiet output

    # --- 3. Assert ---
    print(f"\nIntegration Test RMSE Score (Fast): {rmse}")

    # Assertion: Check for complete model failure
    assert rmse < 1.0, (f"RMSE ({rmse}) is > 1.0. "
                        f"The model is performing very poorly.")