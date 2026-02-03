FROM python:3.9-slim

LABEL maintainer="Benjamin-JHou"
LABEL version="1.0.0"
LABEL description="Hypertension Pan-Comorbidity Multi-Modal Risk Prediction API"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/
COPY README.md .
COPY LICENSE .

# Create directories
RUN mkdir -p logs data/output data/input

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models
ENV CONFIG_PATH=/app/config
ENV LOG_LEVEL=INFO
ENV PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["python", "-m", "src.inference.api_server"]
