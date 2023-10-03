FROM python:3.11.3-slim-buster

RUN pip install torch torchvision torchaudio
RUN pip install opencv-python matplotlib onnx Pillow

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update && apt-get upgrade -y
RUN apt-get install ffmpeg libsm6 libxext6 curl -y

RUN mkdir /api
WORKDIR /api

COPY . .

CMD ["python3", "prediction.py"]
