FROM python:3.8.10

# Prep
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk2.0-dev \
    libboost-all-dev \
    libgl1-mesa-glx && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY setup.py setup.cfg requirements.txt LICENSE MANIFEST.in README.md /app/setup/
COPY mrcnn /app/setup/mrcnn/
WORKDIR /app/setup
RUN pip install -r requirements.txt
RUN pip install . 

# Cleanup
WORKDIR /app
RUN rm -rf /app/setup

# for dev work
COPY . /app

# Set the entrypoint to run your training script
CMD ["tail", "-f", "/dev/null"]
