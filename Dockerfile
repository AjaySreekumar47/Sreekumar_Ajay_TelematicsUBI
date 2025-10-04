# =====================================================================
# Telematics UBI Project - Dockerfile
# =====================================================================

# 1. Base image
FROM python:3.10-slim

# 2. Environment setup
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 3. Working directory
WORKDIR /app

# 4. System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl git \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy project files
COPY requirements.txt .
COPY src/ ./src
COPY bin/ ./bin
COPY docs/ ./docs
COPY data/ ./data
COPY models/ ./models

# 6. Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 7. Expose Streamlit port
EXPOSE 8501

# 8. Default command: run dashboard
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
