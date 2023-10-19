FROM python

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 curl -y

RUN mkdir /ui
WORKDIR /ui

COPY . .

CMD ["python3", "app.py"]
