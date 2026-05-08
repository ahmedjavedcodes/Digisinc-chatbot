# Use a stable Python version
FROM python:3.11-slim

# Install system dependencies for C++ builds
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face MUST use port 7860
# If your file is named app.py, use "app:app". If it is main.py, use "main:app"
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:7860"]