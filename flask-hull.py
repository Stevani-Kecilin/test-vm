import os
import cv2
import numpy as np
from flask import Flask, jsonify, request
import paddlehub as hub
from ultralytics import YOLO
import base64
import time
import requests
import io
from datetime import datetime
import uuid

app = Flask(__name__)

yolo_model = YOLO('hull.pt')  
ocr = hub.Module(name="ch_pp-ocrv3") 

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

EXTERNAL_URL = 'http://dashboard-kpp.kecilin.id/receive_data'

def detect_hull(image):
    results = yolo_model.predict(image)
    bboxes = []
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:  
                bboxes.append(box.xywh.cpu().numpy()[0])
    return bboxes

def draw_bbox(image, bboxes):
    for bbox in bboxes:
        x, y, w, h = bbox
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

def crop_image(image, bbox):
    x, y, w, h = bbox
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)
    return image[y1:y2, x1:x2]

def perform_ocr(image):
    results = ocr.recognize_text(images=[image])
    return results

def save_image(image, filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(file_path, image)
    return file_path

def send_data_to_url(detected_text, saved_images):
    data = {
        "detected_text": detected_text,
        "saved_images": saved_images
    }
    
    try:
        response = requests.post(url=EXTERNAL_URL, json=data, timeout=10)
        if response.status_code not in [200, 201]:
            print(f"Failed to send data: {response.status_code}, {response.text}")
        else:
            print(f"Data sent successfully: {response.status_code}, {response.text}")
    except requests.Timeout:
        print(f"Error: Request to {EXTERNAL_URL} timed out")
    except requests.RequestException as e:
        print(f"Request error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

@app.route('/process_rtsp', methods=['POST'])
def process_rtsp_stream():
    rtsp_url = request.json.get('rtsp_url')

    if not rtsp_url:
        return jsonify({'error': 'No RTSP URL provided'}), 400

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        return jsonify({'error': 'Unable to open RTSP stream'}), 400

    detected_text = []
    saved_images = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  

        frame_count += 1

        if frame_count % 25 == 0:
            bboxes = detect_hull(frame)
            if len(bboxes) > 0:
                frame_with_bboxes = draw_bbox(frame.copy(), bboxes)

                first_bbox = bboxes[0]
                cropped_image = crop_image(frame, first_bbox)
                ocr_results = perform_ocr(cropped_image)

                if ocr_results:
                    for result in ocr_results:
                        for item in result['data']:
                            detected_text.append(item['text'])

                timestamp = int(time.time())  
                filename = f'frame_{frame_count}_{timestamp}.jpg'
                saved_image_path = save_image(frame_with_bboxes, filename)
                saved_images.append(saved_image_path)

                if len(saved_images) >= 15:
                    break

    cap.release()

    send_data_to_url(detected_text, saved_images)

    return jsonify({'message': 'Data processing complete and sent to external URL'}), 200

if __name__ == '__main__':
    app.run(debug=True)
