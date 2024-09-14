FROM tensorflow/tensorflow:2.2.0-gpu

RUN apt-key del 7fa2af80 || true

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get --allow-unauthenticated update 
RUN apt-get install -y \
    software-properties-common \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip \
    gnupg \
    curl

RUN add-apt-repository ppa:deadsnakes/ppa

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

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

RUN python -m pip install --upgrade pip

WORKDIR /app

COPY setup.py setup.cfg requirements.txt LICENSE MANIFEST.in README.md /app/
COPY mrcnn /app/mrcnn/

RUN python -m pip install . 
RUN python -m pip install -r requirements.txt

COPY plantd-transfer-learning/disease_maskrcnn.py /app/

ENV TF_GPU_ALLOCATOR=cuda_malloc_async
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

CMD ["python", "disease_maskrcnn.py"]
