FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (needed for some PyG libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code and necessary project files
COPY backend/ ./backend/
COPY data/ ./data/
COPY results/ ./results/
COPY models/ ./models/

# Expose FastAPI port
EXPOSE 8000

# Start server
CMD ["python", "-m", "uvicorn", "backend.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
