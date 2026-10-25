# Multi-stage production container for ALBUS Quant Research Copilot
FROM python:3.11-slim

# Set working directory and environment variables
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

# Install minimal system dependencies for building C-extensions and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging specifications first for caching
COPY pyproject.toml ./

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy project source and configuration
COPY config/ ./config/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY results/ ./results/

# Expose HTTP health check port (8080) and Streamlit UI port (8501)
EXPOSE 8080
EXPOSE 8501

# Health check probe against /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Launch health server and Streamlit app
CMD python src/health_server.py & streamlit run src/app.py --server.port=8501 --server.address=0.0.0.0
