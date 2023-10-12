FROM ultralytics/ultralytics

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN mkdir /api
WORKDIR /api

COPY . .

CMD ["python3", "prediction.py"]
