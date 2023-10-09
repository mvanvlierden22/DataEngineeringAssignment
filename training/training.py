from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    model.train(data="/app/datasets/custom_data.yaml", epochs=3, name="yolov8n_custom")
    path = model.export()  # export the model to ONNX format


if __name__ == "__main__":
    main()
