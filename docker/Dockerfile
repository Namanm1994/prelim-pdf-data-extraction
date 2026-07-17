FROM python:3.11-slim

# System dependencies
# - tesseract-ocr: OCR engine
# - poppler-utils: pdf2image dependency (pdftoppm)
# - libgl1: OpenCV dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying source
# Keeps this layer cached unless requirements change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Output directory (mount a volume here in production)
RUN mkdir -p /app/output

CMD ["python", "main.py"]
