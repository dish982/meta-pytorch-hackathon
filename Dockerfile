
FROM python:3.10-slim

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/huggingface \
    ENABLE_WEB_INTERFACE=true

WORKDIR /app

# Install git (needed for OpenEnv install)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Permissions
RUN chmod -R 777 /app

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]