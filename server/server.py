# server.py - chạy trên máy tính nhận ảnh
from flask import Flask, request
import os, time

app = Flask(__name__)
SAVE_DIR = "received_images"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No image", 400

    file = request.files['image']
    detections = request.form.get('detections', '')
    
    # Luu anh xuong thu muc
    filepath = os.path.join(SAVE_DIR, file.filename)
    file.save(filepath)
    
    print(f"[NHAN] {file.filename}")
    print(f"[BENH] {detections}")
    return "OK", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)