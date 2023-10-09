from ultralytics import YOLO
import argparse
from google.cloud import storage
import shutil


def train_yolo(project_id, data_bucket, model_repo):
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    dataset_path = data_bucket + "/data.yaml"
    model.train(data=dataset_path, epochs=3, name="yolov8n_custom")

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
    parser.add_argument("--dataset_path", type=str, help="CSV file with features")
    parser.add_argument("--model_repo", type=str, help="Name of the model bucket")
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    train_yolo(**parse_command_line_arguments())
