from flask import Flask, request, jsonify
import cv2
import numpy as np
import time
import requests
from ultralytics import YOLO
import os

# ================== CONFIG ==================
MODEL_PATH = "yolov8n.pt"

BOT_TOKEN = "8500551600:AAHaYZZFuDgHV5qlTCCBi2x2ldyEjgYuPBc"
CHAT_ID = "6863875327"

FALL_TIME_THRESHOLD = 2  # giây
fall_start_time = None

# ================== INIT ==================
app = Flask(__name__)
model = YOLO(MODEL_PATH)

# ================== TELEGRAM ==================
def send_telegram_text(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": msg})

def send_telegram_photo(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    requests.post(
        url,
        data={"chat_id": CHAT_ID},
        files={"photo": buffer.tobytes()}
    )

# ================== API ==================
@app.route("/upload", methods=["POST"])
def upload():
    global fall_start_time

    # ESP32 gửi RAW JPEG
    data = request.data
    if not data:
        return "NO DATA", 400

    img_bytes = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return "BAD IMAGE", 400

    results = model(frame, conf=0.4)

    person_detected = False
    fall_detected = False
    status_text = "NO PERSON"

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            if cls_name == "person":
                person_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                ratio = w / h

                if ratio > 1.2:
                    status_text = "LYING"
                    if fall_start_time is None:
                        fall_start_time = time.time()
                    elif time.time() - fall_start_time > FALL_TIME_THRESHOLD:
                        fall_detected = True
                        status_text = "FALL DETECTED"
                else:
                    status_text = "STANDING"
                    fall_start_time = None

    if fall_detected:
        print(">>> FALL DETECTED <<<")
        send_telegram_text("🚨 KHẨN CẤP: PHÁT HIỆN NGƯỜI BỊ NGÃ!")

    return jsonify({
        "person": person_detected,
        "fall": fall_detected,
        "status": status_text
    }), 200

# ================== RUN ==================
if __name__ == "__main__":
    print("🚀 Server AI đang chạy tại http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)
@app.route("/", methods=["GET"])
def index():
    return "AI SERVER RUNNING", 200
