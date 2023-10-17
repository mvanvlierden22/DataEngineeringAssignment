from ultralytics import YOLO, settings


def main():
    settings.update({"datasets_dir": "null"})
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    model.train(
        data="obj_det_dataset/data.yaml", epochs=3, name="yolov8n_custom", device="cpu"
    )
    path = model.export()  # export the model to ONNX format


if __name__ == "__main__":
    main()
