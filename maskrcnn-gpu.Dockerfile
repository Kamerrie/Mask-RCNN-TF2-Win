# Start from the TensorFlow 2.2.0 GPU image
FROM tensorflow/tensorflow:2.2.0-gpu

# Removing dead gpg keys
RUN apt-key del 7fa2af80 || true

# get new keys
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# Install dependencies for Python 3.8.10
RUN apt-get --allow-unauthenticated update 
RUN apt-get install -y \
    software-properties-common \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip \
    gnupg \
    curl

# Add the deadsnakes PPA for Python 3.8
RUN add-apt-repository ppa:deadsnakes/ppa

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    python3.8-venv \
    build-essential \
    cmake \
    git \
    curl \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk2.0-dev \
    libboost-all-dev \
    libgl1-mesa-glx \
    wget \
    unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Update alternatives to use Python 3.8
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Upgrade pip for Python 3.8
RUN python -m pip install --upgrade pip

# Set the working directory
WORKDIR /app

# Copy project files
COPY setup.py setup.cfg requirements.txt LICENSE MANIFEST.in README.md /app/
COPY mrcnn /app/mrcnn/

# Install Python dependencies
RUN python -m pip install . 
RUN python -m pip install -r requirements.txt

# Copy this at the end to prevent redoing pip installs.
COPY plantd-transfer-learning/disease_maskrcnn.py /app/

# Set the entrypoint to run your training script
CMD ["python", "disease_maskrcnn.py"]
