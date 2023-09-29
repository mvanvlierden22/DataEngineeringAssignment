from ultralytics import YOLO


def main():
    model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)
    model.train(data="coco128.yaml", epochs=3)
    path = model.export()  # export the model to ONNX format


if __name__ == "__main__":
    main()
