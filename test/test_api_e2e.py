# test/test_api_e2e.py

import pytest
import requests
import subprocess
import time
import os
from pathlib import Path

# === Test Configuration ===
PROJECT_ROOT = Path(__file__).parent.parent
IMAGE_NAME = "recsys-api:latest"
CONTAINER_NAME = "test_recsys_api_service"
API_URL = "http://127.0.0.1:8001"
HEALTH_CHECK_URL = f"{API_URL}/"
PREDICT_URL = f"{API_URL}/recommend/"
MODEL_PATH = PROJECT_ROOT / "models"


@pytest.fixture(scope="module")
def api_service():
    """
    pytest Fixture: Manages the lifecycle of the v3.1 RecSys API container.
    """

    # --- Setup ---
    print(f"\n[Setup] Starting v3.1 API Docker container '{IMAGE_NAME}'...")

    model_vol = f"{PROJECT_ROOT.resolve()}/models:/app/models"
    processed_vol = f"{PROJECT_ROOT.resolve()}/data/processed:/app/data/processed"
    raw_vol = f"{PROJECT_ROOT.resolve()}/data/raw:/app/data/raw"

    start_command = [
        "docker", "run",
        "-d", "--rm",
        "--name", CONTAINER_NAME,
        "-p", "8001:80",
        "-v", model_vol,
        "-v", processed_vol,
        "-v", raw_vol,
        IMAGE_NAME
    ]

    try:
        subprocess.run(start_command, check=True, capture_output=True)
        print(f"[Setup] Container '{CONTAINER_NAME}' started on port 8001.")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Failed to start Docker container. Is the '{IMAGE_NAME}' "
                    f"image built? Is Docker running? Error: {e.stderr.decode()}")

    # Wait for the Uvicorn server (and 3-file-load) to boot (15 seconds)
    time.sleep(15)

    # --- Health Check ---
    retries = 5
    for i in range(retries):
        try:
            response = requests.get(HEALTH_CHECK_URL, timeout=5)
            if response.status_code == 200:
                print("[Setup] Health check passed. API is live.")
                break
            time.sleep(2)
        except requests.exceptions.ConnectionError:
            if i == retries - 1:
                pytest.fail("E2E Test Failed: Could not connect to the API.")
            time.sleep(2)

    yield PREDICT_URL

    # --- Teardown ---
    print(f"\n[Teardown] Stopping container '{CONTAINER_NAME}'...")
    subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True)
    print("[Teardown] Container stopped and removed.")


@pytest.mark.slow
def test_api_recommend_endpoint_success(api_service):
    """
    Test v5.3 (E2E Test):
    Sends a valid UserID (User 1) to the live API
    and asserts a 200 (OK) response with the correct structure.
    """
    # Arrange: User 1 exists
    user_id = 1

    # Act
    response = requests.get(f"{api_service}{user_id}")

    # Assert (Robust Tests - Not Brittle)
    assert response.status_code == 200
    data = response.json()
    assert data["UserID"] == user_id
    assert len(data["Recommendations"]) == 10  # TOP_K_RECOMMENDATIONS
    assert "Title" in data["Recommendations"][0]
    #
    # === "KAOS" (Brittle Test) REMOVED ===
    # assert "Star Wars" in data["Recommendations"][0]["Title"]
    # We no longer assert a *specific* movie,
    # as the model (GridSearchCV) may change the order.
    # We only assert that the *structure* (Title) is correct.
    #


@pytest.mark.slow
def test_api_recommend_endpoint_user_not_found(api_service):
    """
    Test v5.3 (E2E Test):
    Sends an invalid UserID (999999) to the live API
    and asserts that the API correctly returns a 404 (Not Found) error.
    """
    # Arrange: User 999999 does not exist
    user_id = 999999

    # Act
    response = requests.get(f"{api_service}{user_id}")

    # Assert
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "User ID 999999 not found" in data["detail"]