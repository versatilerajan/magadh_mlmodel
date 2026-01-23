FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY train.py .
COPY app.py .

# Train models during build
RUN python train.py

# Verify models exist
RUN ls -la models/

# Create necessary directories
RUN mkdir -p /tmp/uploads

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Expose port
EXPOSE 5000

# Use exec form and simpler gunicorn config
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "300", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
