# app/main.py

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from app.schema import PredictionResponse, Movie
from typing import List
import sys

# --- Our Project Imports (The "Google-level" reward) ---
# We can import from 'src' because of 'pip install -e .'
from src.config import (
    MODEL_OUTPUT_PATH,
    MOVIES_CLEAN_PATH,  # The "polish" file (movies_cleaned.pkl)
    TOP_K_RECOMMENDATIONS,
    RATINGS_DATA_PATH,  # We need this to know what the user *already* watched
    RATINGS_COLS
)
from src.data_processing import DATA_SEPARATOR, DATA_ENGINE  # Need these to read ratings.dat

# --- Application and Model Loading ---

app = FastAPI(
    title="Movie Recommendation API",
    description="v3.0 - A 'Google-level' MLOps API for SVD-based recommendations.",
    version="3.0.0"
)


# This is our "state". We load all 3 critical files into memory
# at startup, so the API is fast.
class ModelCache:
    model = None  # The SVD Algorithm (recsys_svd_model.pkl)
    movies_df = None  # The Movie Titles (movies_cleaned.pkl)
    user_watch_list = None  # A dict of movies users have *already* rated


app.state.cache = ModelCache()


@app.on_event("startup")
def load_model_and_data():
    """
    API Startup Event:
    Load the SVD model, the clean movie data (for "polish"),
    and the raw ratings data (to prevent re-recommending).
    """
    print("API is starting up...")

    # 1. Load the SVD Model (v2.3)
    try:
        app.state.cache.model = joblib.load(MODEL_OUTPUT_PATH)
        print(f"Model loaded from: {MODEL_OUTPUT_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_OUTPUT_PATH}.")
        sys.exit(1)  # Fail fast

    # 2. Load the Clean Movies Data (v3.0 "Polish")
    try:
        app.state.cache.movies_df = pd.read_pickle(MOVIES_CLEAN_PATH)
        # Set MovieID as the index for fast lookups (e.g., .loc[1196])
        app.state.cache.movies_df.set_index("MovieID", inplace=True)
        print(f"Clean movie titles loaded from: {MOVIES_CLEAN_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Clean movies file not found at {MOVIES_CLEAN_PATH}.")
        print("Please run 'python -m src.train' first to create it.")
        sys.exit(1)

    # 3. Load Raw Ratings (to know what users *already* watched)
    try:
        ratings_df = pd.read_csv(
            RATINGS_DATA_PATH,
            sep=DATA_SEPARATOR,
            header=None,
            names=RATINGS_COLS,
            engine=DATA_ENGINE
        )
        # Create a dict: {UserID: set(MovieIDs they rated)}
        # This is a "Google-level" optimization for fast lookups.
        app.state.cache.user_watch_list = ratings_df.groupby('UserID')['MovieID'].apply(set)
        print("User watch list created.")

    except FileNotFoundError:
        print(f"ERROR: Raw ratings file not found at {RATINGS_DATA_PATH}.")
        sys.exit(1)

    print("API startup complete. Model and data are loaded.")


# --- API Endpoints ---

@app.get("/", tags=["Health Check"])
def read_root():
    """Root endpoint for health checks."""
    if app.state.cache.model and app.state.cache.movies_df is not None:
        return {"status": "ok", "message": "Recommendation API is running!"}
    return {"status": "error", "message": "Model or data is not loaded!"}


@app.get("/recommend/{user_id}",
         response_model=PredictionResponse,
         tags=["Recommendation"])
def get_recommendations(user_id: int):
    """
    v3.0 - The main recommendation endpoint.

    Takes a UserID and returns the Top-K movie recommendations,
    filtering out movies the user has already watched.
    """
    cache = app.state.cache

    # "Kaos" (Error) Handling: Does this user exist?
    if user_id not in cache.user_watch_list:
        raise HTTPException(
            status_code=404,
            detail=f"User ID {user_id} not found in the ratings data."
        )

    if cache.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    # --- "Google-level" Inference Logic ---

    # 1. Get the set of movies the user has *already* watched
    watched_movies = cache.user_watch_list.get(user_id, set())

    # 2. Get the list of *all* MovieIDs (from the "polish" file)
    all_movie_ids = cache.movies_df.index.tolist()

    # 3. Create a list of movies to *predict*
    # (i.e., all movies user has *not* watched)
    movies_to_predict = [
        movie_id for movie_id in all_movie_ids
        if movie_id not in watched_movies
    ]

    print(f"Predicting ratings for {len(movies_to_predict)} movies for User {user_id}...")

    # 4. Run the SVD model (algo.predict) for every un-watched movie
    # This is the "kaos" of the 'surprise' library:
    # algo.predict() takes (uid, iid)
    predictions = [
        (movie_id, cache.model.predict(uid=user_id, iid=movie_id).est)
        for movie_id in movies_to_predict
    ]

    # 5. Sort the predictions by the estimated rating (highest first)
    predictions.sort(key=lambda x: x[1], reverse=True)

    # 6. Get the Top-K recommendations
    top_k_preds = predictions[:TOP_K_RECOMMENDATIONS]
    top_k_movie_ids = [movie_id for (movie_id, rating) in top_k_preds]

    # 7. "Polish" the response: Convert MovieIDs to Titles
    # (Using the 'movies_df' we loaded at startup)
    top_k_movies_df = cache.movies_df.loc[top_k_movie_ids]

    # Convert the DataFrame rows to our Pydantic 'Movie' objects
    recommended_movies = [
        Movie(
            MovieID=row.Index,
            Title=row.Title,
            Genres=row.Genres
        ) for row in top_k_movies_df.itertuples()
    ]

    # 8. Return the "Google-level" response
    return {
        "UserID": user_id,
        "Recommendations": recommended_movies
    }