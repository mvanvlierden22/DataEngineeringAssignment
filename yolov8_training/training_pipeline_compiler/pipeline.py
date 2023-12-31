from kfp import compiler
import kfp
import typing
from typing import Dict
from typing import NamedTuple
from kfp import dsl
from kfp.dsl import (
    Artifact,
    Dataset,
    Input,
    Model,
    Output,
    Metrics,
    ClassificationMetrics,
    component,
    OutputPath,
    InputPath,
)
import os
import logging

logging.basicConfig(level=logging.DEBUG)

# The Google Cloud project that this pipeline runs in.
PROJECT_ID = "data-engineering-assignment-1"
# The region that this pipeline runs in
REGION = "europe-west1"
# Specify a Cloud Storage URI that your pipelines service account can access. The artifacts of your pipeline runs are stored within the pipeline root.
PIPELINE_ROOT = "gs://tmp_data_engineering_assignment_group_7"

# Load TAG from environment variable
TAG_NAME = os.environ.get("TAG_NAME")


@dsl.container_component
def downloading(
    project_id: str, data_repo: str, data_file_name: str, dataset_path: Output[Artifact]
):
    return dsl.ContainerSpec(
        image=f"europe-west1-docker.pkg.dev/data-engineering-assignment-1/docker-image-repo/yolov8_downloader:{TAG_NAME}",
        command=["python3", "/pipelines/component/src/component.py"],
        args=[
            "--project_id",
            project_id,
            "--data_bucket",
            data_repo,
            "--file_name",
            data_file_name,
            "--dataset_path",
            dataset_path.path,
        ],
    )


@dsl.container_component
def training(project_id: str, dataset_path: Input[Artifact], model_repo: str):
    return dsl.ContainerSpec(
        image=f"europe-west1-docker.pkg.dev/data-engineering-assignment-1/docker-image-repo/yolov8_trainer:{TAG_NAME}",
        command=["python3", "/pipelines/component/src/component.py"],
        args=[
            "--project_id",
            project_id,
            "--dataset_path",
            dataset_path.path,
            "--model_repo",
            model_repo,
        ],
    )


# Define the workflow of the pipeline.
@kfp.dsl.pipeline(
    name="yolov8-training-pipeline",
    description="A yolov8 training pipeline",
    pipeline_root=PIPELINE_ROOT,
)
def pipeline(project_id: str, data_bucket: str, model_repo: str, data_file_name: str):
    # The first step
    download_op = downloading(
        project_id=project_id, data_repo=data_bucket, data_file_name=data_file_name
    )

    # The second step
    training_op = training(
        project_id=project_id,
        dataset_path=download_op.outputs["dataset_path"],
        model_repo=model_repo,
    )


def list_all_files():
    path = "./"

    # We shall store all the file names in this list
    filelist = []

    for root, dirs, files in os.walk(path):
        for file in files:
            # append the file name to the list
            filelist.append(os.path.join(root, file))

    # Print all the file names
    for name in filelist:
        logging.info(name)


if __name__ == "__main__":
    logging.info("Compiling pipeline...")
    compiler.Compiler().compile(
        pipeline_func=pipeline, package_path="yolov8-training-pipeline.yaml"
    )
    logging.info("Pipeline compiled!")
    list_all_files()
