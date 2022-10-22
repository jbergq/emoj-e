# Build:
# sudo docker build -t <project_name> .. -f Dockerfile
# 
# Run:
# sudo docker run -v /mnt/nas:/mnt/nas --gpus all -it <project_name>

FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu18.04

ENV PYTHON_VERSION=3.8
ENV DEBIAN_FRONTEND="noninteractive"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONPATH="${PYTHONPATH}:/workspace/ext/BLIP"

ARG UID=1500
ARG GID=1500
ARG USERNAME=test

# Install dependencies.
RUN apt update
RUN apt-get update

RUN apt install -y bash \
    build-essential \
    ca-certificates \
    wget

# Create python 3.8 venv
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes -y
RUN apt-get install python3.8-venv python3-pip python3.8-dev -y

RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

RUN apt-get install git rsync -y
RUN apt-get install docker.io -y
RUN pip3 install -U pip

# Install packages needed for opencv.
RUN apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /workspace/ext/BLIP

# Copy requirements file to container and install packages.
# COPY ext/BLIP ext/BLIP
# COPY caption_images.py .
COPY ext/BLIP/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install setuptools==59.5.0

CMD python ../../caption_images.py -i /workspace/datasets/slackmojis/train/ -o /workspace/test/
