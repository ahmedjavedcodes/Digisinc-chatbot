# Use a stable Python version
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Install system dependencies
# These are required for compiling certain Python packages like scikit-learn
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and set the working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir keeps the final image size smaller
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port (Hugging Face uses 7860)
EXPOSE 7860

# Start the application
# We use 'python -m gunicorn' for better reliability in finding the executable
# If your file is named 'main.py', change 'app:app' to 'main:app'
CMD ["python", "-m", "gunicorn", \
    "-w", "4", \
    "-k", "uvicorn.workers.UvicornWorker", \
    "app:app", \
    "--bind", "0.0.0.0:7860", \
    "--timeout", "120"]