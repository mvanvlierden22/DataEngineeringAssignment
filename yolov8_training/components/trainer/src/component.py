from ultralytics import YOLO, settings
import argparse
from google.cloud import storage
import shutil
import os
import zipfile
import sys


def train_yolo(project_id, dataset_path, model_repo):
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall("./")

    print([x[0] for x in os.walk("./")])
    print(os.getcwd())

    dataset_folder = "obj_det_dataset"

    # Make sure yolo knows where to find dataset
    settings.update({"datasets_dir": "null"})

    yaml = f"{dataset_folder}/data.yaml"
    model.train(data=yaml, epochs=1, name="yolov8n_custom", device="cpu")

    local_file = "runs/detect/yolov8n_custom/weights/best.pt"
    # Save to GCS
    client = storage.Client(project=project_id)
    bucket = client.get_bucket(model_repo)
    blob = bucket.blob("yolov8_custom.pt")
    # Upload local file
    blob.upload_from_filename(local_file)
    # Clean up
    shutil.rmtree("runs")


# Defining and parsing the command-line arguments
def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=str, help="GCP project id")
    parser.add_argument("--dataset_path", type=str, help="Dataset path")
    parser.add_argument("--model_repo", type=str, help="Name of the model bucket")
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    train_yolo(**parse_command_line_arguments())
