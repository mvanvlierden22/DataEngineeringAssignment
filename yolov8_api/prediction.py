from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
from PIL import Image
from io import BytesIO
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import torch
import os
import uvicorn
from ultralytics import YOLO


# Create the FastAPI app
app = FastAPI()

# load a pretrained model for inference (will be replaced by trained models when training pipeline has been set up )
model = YOLO("yolov8s.pt")


@app.post("/detect")
async def detect(file: UploadFile):
    path = f"/app/tmp/{file.filename}"
    with open(path, "wb") as f:
        contents = await file.read()
        f.write(contents)
    results = model(path)  # predict on posted image
    os.remove(path)
    return JSONResponse(results[0].tojson())


if __name__ == "__main__":
    uvicorn.run(
        "prediction:app",
        host="0.0.0.0",
        port=5000,
        reload=False,
        workers=5,
    )
