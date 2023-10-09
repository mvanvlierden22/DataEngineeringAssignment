FROM python:3.11.3-slim-buster

WORKDIR /pipelines/component/src
ADD src/component.py ./

RUN pip install torch torchvision torchaudio
RUN pip install opencv-python matplotlib onnx Pillow ultralytics

RUN apt-get update && apt-get upgrade -y
RUN apt-get install ffmpeg libsm6 libxext6 curl -y

