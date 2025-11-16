# test/test_pipeline.py

import pytest
import pandas as pd

# Import the functions to be tested from the source code
from src.data_processing import load_ratings_data, load_and_save_movies_data
from src.config import RATINGS_COLS, MOVIES_COLS


# Note: This project has no 'pipeline.py',
# so we unit-test 'data_processing.py' instead.

def test_load_ratings_data_structure():
    """
    Test v5.1 (Unit Test):
    Validates that load_ratings_data() returns a DataFrame
    with the correct columns defined in config.py.
    """
    # Act
    try:
        df = load_ratings_data()
    except FileNotFoundError:
        pytest.skip("Skipping test: Raw 'ratings.dat' file not found.")

    # Assert
    assert isinstance(df, pd.DataFrame)
    # This tests the 'data_processing' logic
    assert list(df.columns) == ["UserID", "MovieID", "Rating"]


def test_load_movies_data_structure():
    """
    Test v5.1 (Unit Test):
    Validates that load_and_save_movies_data() returns a DataFrame
    with the correct columns defined in config.py.
    """
    # Act
    try:
        df = load_and_save_movies_data()
    except FileNotFoundError:
        pytest.skip("Skipping test: Raw 'movies.dat' file not found.")

    # Assert
    assert isinstance(df, pd.DataFrame)
    # This tests the 'config.py' and 'data_processing.py' consistency
    assert list(df.columns) == MOVIES_COLS