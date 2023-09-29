from fastapi import FastAPI, File, UploadFile, Request
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
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware


# Create the FastAPI app
app = FastAPI()

# API Key
api_key = os.getenv("POTREE_API_TOKEN")

# load a pretrained model for inference (will be replaced by trained models when training pipeline has been set up )
model = YOLO("yolov8s.pt")


@app.middleware("http")
async def validate_api_key(request, call_next):
    root_path = request.scope.get("root_path", "").rstrip("/")
    if request.url.path.startswith(
        root_path + "/openapi"
    ) or request.url.path.startswith(root_path + "/docs"):
        return await call_next(request)

    provided_api_key = request.headers.get("Authorization")
    if provided_api_key != api_key:
        return JSONResponse(status_code=401, content={"message": "Not authenticated!"})

    response = await call_next(request)
    return response


# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(req: Request):
    root_path = req.scope.get("root_path", "").rstrip("/")
    openapi_url = root_path + app.openapi_url
    return get_swagger_ui_html(
        openapi_url=openapi_url,
        title="API",
    )


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
