from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from ultralytics import YOLO
import os

MODEL_DIR = 'weights'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
model_weight = 'yolov8n.pt'
model_path = os.path.join(MODEL_DIR, model_weight)

app = FastAPI()

model = YOLO(model_path)

def process_count_people(img):
    results = model.predict(
                source=img,
                classes=[0],
                verbose=False,               
            )
    
    bboxes = results[0].boxes.xyxy.int().tolist()
    bboxes = [{"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]} for bbox in bboxes]

    conf = results[0].boxes.conf.tolist()
    conf = [round(c, 2) for c in conf]

    return {
        "bbox": bboxes,
        "conf": conf,
        "count": len(bboxes)
    }
    
@app.post("/count_people/")
async def count_people(image_file: UploadFile = File(...)):
    # Read the image
    contents = await image_file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process YOLO
    data = process_count_people(img)

    return data

if __name__ == "__main__":
    img = cv2.imread("test.jpg")
    data = process_count_people(img)

    print(data)
