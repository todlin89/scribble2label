FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt tifffile

# Copy project source code
COPY scripts/ scripts/
COPY Inference_custom.py .

# Copy model weights
COPY logs/fiji_BC_tile512/best_model.pth /app/model/best_model.pth

ENTRYPOINT ["python", "Inference_custom.py"]
CMD ["--input_dir", "/data/input", "--output_dir", "/data/output", "--model_path", "/app/model/best_model.pth"]
