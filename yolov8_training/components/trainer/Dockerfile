FROM ultralytics/ultralytics

WORKDIR /pipelines/component/src
ADD src/component.py ./
ADD src/settings.yaml /root/.config/Ultralytics/settings.yaml

RUN pip install google-cloud-storage
