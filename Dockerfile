FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    vim \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    matplotlib==3.7.1 \
    pyyaml==6.0 \
    pillow==9.5.0 \
    torchvision==0.15.1

# Copy files
COPY config.yaml generator.py discriminator.py main.py helper.py /app/
COPY README.md /app/

# Create results directory
RUN mkdir -p /app/results

# Set default command
CMD ["python", "main.py"]