# v3.1 API Server Dockerfile (v4 Strategy: Pip-Optimized + GCC Fix)

# Step 1: Base Image
FROM python:3.10-slim-bookworm

# Step 2: Set the Working Directory
WORKDIR /app

# Step 3: Copy requirements
COPY requirements.txt requirements.txt

# Step 4: Install Build Dependencies ("The Kaos Fix")
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Step 5: Install Python Dependencies
RUN pip install -r requirements.txt

# Step 6: Install Project Package
COPY setup.py setup.py
RUN pip install .

# Step 7: Copy All Project Code
COPY . .

# Step 8: Expose API Port
EXPOSE 80

# Step 9: Define Start Command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
