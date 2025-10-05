# Use Python 3.9.6 base image
FROM python:3.9.6-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (optional, in case numpy/scipy need them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
COPY . /app

# Ensure the app folder is in Python path (so "import app" works)
ENV PYTHONPATH=/app

# Expose port (change if your app runs on a different port)
EXPOSE 5000

# Default command to run your FastAPI/Flask/whatever app
# Adjust if you use gunicorn/uvicorn instead of python
CMD ["python", "main.py"]
