import cv2
import numpy as np
import time
import os
import threading
import requests
from tflite_runtime.interpreter import Interpreter

# --- 1. CẤU HÌNH ---
MODEL_PATH = "yolov8n_benh_la_int8.tflite"
LABEL_PATH = "benh_la2.txt"
SERVER_URL = "http://192.168.1.100:5000/upload"
COOLDOWN_TIME = 5.0

if os.path.exists(LABEL_PATH):
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Đã nạp {len(labels)} nhãn: {labels}")
else:
    print("CẢNH BÁO: Không tìm thấy file label, dùng nhãn mặc định.")
    labels = ["Class_0", "Class_1", "Class_2", "Class_3"]

CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.4
SKIP_FRAMES = 4
last_upload_times = {label: 0.0 for label in labels}

# --- 2. HÀM GỬI SERVER ---
def upload_to_server(frame, label_name, score_percent):
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        files = {'image': (f'{label_name}.jpg', img_encoded.tobytes(), 'image/jpeg')}
        data = {'label': label_name, 'confidence': score_percent, 'timestamp': time.time()}
        response = requests.post(SERVER_URL, files=files, data=data, timeout=5)
        print(f"[SERVER] Đã gửi {label_name} - Status: {response.status_code}")
    except Exception as e:
        print(f"[SERVER LỖI] {e}")

# --- 3. NẠP MODEL ---
print("Đang nạp YOLOv8 TFLite...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
MODEL_H = input_details[0]['shape'][1]
MODEL_W = input_details[0]['shape'][2]
input_scale, input_zero_point = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']

# --- 4. CAMERA ---
class ThreadedCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.raw_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.raw_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

print("Khởi động camera...")
cap = ThreadedCamera(src=0).start()
x_scale = cap.raw_w / MODEL_W
y_scale = cap.raw_h / MODEL_H

frame_count = 0
fps_avg = 0
start_time = time.time()
last_results = []

print("Bắt đầu quét bệnh. Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1
    display_frame = frame.copy()

    # --- SKIP FRAMES ---
    if frame_count % SKIP_FRAMES != 0:
        for box_info in last_results:
            x, y, w, h, label_name, score_percent = box_info
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.rectangle(display_frame, (x, y-25), (x + len(label_name)*12 + 40, y), (0, 255, 0), -1)
            cv2.putText(display_frame, f"{label_name} {score_percent}%", (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(display_frame, f"FPS: {fps_avg:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('YOLOv8 TFLite', display_frame)
        if cv2.waitKey(1) == ord('q'): break
        continue

    # --- INFERENCE ---
    t_inf_start = time.time()
    img_resized = cv2.resize(frame, (MODEL_W, MODEL_H))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    if input_details[0]['dtype'] == np.int8:
        input_data = (np.float32(img_rgb) / 255.0) / input_scale + input_zero_point
        input_data = np.clip(input_data, -128, 127).astype(np.int8)
    else:
        input_data = np.float32(img_rgb) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    if output_details[0]['dtype'] == np.int8 and output_scale > 0:
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

    predictions = np.squeeze(output_data)  # bỏ .T

    boxes_list, confidences, class_ids = [], [], []
    for row in predictions:
        classes_scores = row[4:]
        class_id = np.argmax(classes_scores)
        score = classes_scores[class_id]
        if score > CONF_THRESHOLD:
            xc, yc, w, h = row[0], row[1], row[2], row[3]
            # scale từ normalized sang input size
            xc *= MODEL_W
            yc *= MODEL_H
            w  *= MODEL_W
            h  *= MODEL_H
            x_min = xc - w/2
            y_min = yc - h/2
            left   = max(0, int(x_min * x_scale))
            top    = max(0, int(y_min * y_scale))
            width  = int(w * x_scale)
            height = int(h * y_scale)
            boxes_list.append([left, top, width, height])
            confidences.append(float(score))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes_list, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    last_results.clear()

    if len(indices) > 0:
        current_time = time.time()
        for i in indices.flatten():
            x, y, w, h = boxes_list[i]
            label_id = class_ids[i]
            score_percent = int(confidences[i] * 100)
            label_name = labels[label_id] if 0 <= label_id < len(labels) else f"Class_{label_id}"
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.rectangle(display_frame, (x, y-25), (x + len(label_name)*12 + 40, y), (0, 255, 0), -1)
            cv2.putText(display_frame, f"{label_name} {score_percent}%", (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            last_results.append([x, y, w, h, label_name, score_percent])
            if current_time - last_upload_times.get(label_name, 0) > COOLDOWN_TIME:
                threading.Thread(target=upload_to_server,
                                 args=(display_frame.copy(), label_name, score_percent)).start()
                last_upload_times[label_name] = current_time

    inf_time = (time.time() - t_inf_start) * 1000
    fps_avg = 1 / (time.time() - start_time) * SKIP_FRAMES
    start_time = time.time()
    cv2.putText(display_frame, f"FPS: {fps_avg:.1f} | AI: {inf_time:.0f}ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('YOLOv8 TFLite', display_frame)
    if cv2.waitKey(1) == ord('q'): break

cap.stop()
cv2.destroyAllWindows()
print("Đã thoát chương trình.")