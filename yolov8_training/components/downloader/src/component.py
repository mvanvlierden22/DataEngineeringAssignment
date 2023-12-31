from google.cloud import storage
import sys
import logging
from pathlib import Path
import zipfile
import argparse
import os


def download_data(project_id, data_bucket, file_name, dataset_path):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    client = storage.Client(project=project_id)
    bucket = client.get_bucket(data_bucket)
    blob = bucket.blob(file_name)
    # Creating the directory where the output file is created (the directory
    # may or may not exist).
    Path(dataset_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(dataset_path)

    print([x[0] for x in os.walk("./")])

    logging.info("Downloaded Data!")


# Defining and parsing the command-line arguments
def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=str, help="GCP project id")
    parser.add_argument("--data_bucket", type=str, help="Name of the data bucket")
    parser.add_argument(
        "--file_name", type=str, help="Name of the training data set file name"
    )
    parser.add_argument("--dataset_path", type=str, help="Path to dataset")
    args = parser.parse_args()
    return vars(
        args
    )  # The vars() method returns the __dict__ (dictionary mapping) attribute of the given object.


if __name__ == "__main__":
    download_data(**parse_command_line_arguments())
