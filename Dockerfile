FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    fonts-liberation \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install --upgrade build
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create directories that will be mounted as volumes
RUN mkdir -p uploads output

EXPOSE 6800

CMD ["python3", "web_server.py"]
