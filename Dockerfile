FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY train.py .
COPY app.py .

# Train models during build
RUN python train.py

# Create necessary directories
RUN mkdir -p /tmp/uploads

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV WORKERS=4
ENV TIMEOUT=300
ENV MAX_REQUESTS=1000
ENV MAX_REQUESTS_JITTER=50
