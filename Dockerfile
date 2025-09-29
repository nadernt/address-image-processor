FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /

# Copy requirements and install
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy models and code
COPY models/ models/
COPY app.py .

# Start the container
CMD ["python", "-u", "app.py"]


