from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2

from analyzer.analyzer import CDPAnalyzer

app = FastAPI()

analyzer = CDPAnalyzer()


@app.get("/")
def root():

    return {
        "status": "running"
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...)
):

    data = await file.read()

    img = cv2.imdecode(
        np.frombuffer(data, np.uint8),
        cv2.IMREAD_COLOR
    )

    result = analyzer.analyze(img)

    return result
