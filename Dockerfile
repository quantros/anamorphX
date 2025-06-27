# 🏢 AnamorphX Enterprise Neural Server - Docker Image
FROM python:3.11-slim

LABEL maintainer="AnamorphX Team <team@anamorph.ai>"
LABEL description="Enterprise Neural Web Server with Backend/Frontend Separation"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ANAMORPH_HOST=0.0.0.0
ENV ANAMORPH_PORT=8080

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs frontend/dist models data

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash anamorph && \
    chown -R anamorph:anamorph /app

USER anamorph

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Default command
CMD ["python3", "enterprise_neural_server.py", "--host", "0.0.0.0", "--port", "8080"] 