import cv2
import numpy as np
import time
import os
import requests  # <-- THEM DE GUI SERVER
from tflite_runtime.interpreter import Interpreter
import threading

# --- 1. CAU HINH HE THONG ---
MODEL_PATH = "D:\Unarrage\benh_la\2. second\yolov8n_benh_la_int8.tflite"
LABEL_PATH = "D:\Unarrage\benh_la\2. second\benh_la2.txt"

# =========================================================
# CAU HINH SERVER - SUA LAI URL CUA BAN
SERVER_URL = "http://192.168.2.25:8080/upload"  # <-- SUA LAI
SEND_COOLDOWN = 3  # Chi gui toi da 1 anh moi 3 giay (tranh spam)
last_sent_time = 0
# =========================================================

if os.path.exists(LABEL_PATH):
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Da nap {len(labels)} nhan tu {LABEL_PATH}: {labels}")
else:
    print(f"CANH BAO: Khong tim thay {LABEL_PATH}. Se dung nhan mac dinh.")
    labels = ["Class_0", "Class_1", "Class_2", "Class_3"]

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
SKIP_FRAMES = 4

# --- 2. NAP MO HINH TFLITE ---
print("Dang nap mo hinh YOLOv8 TFLite...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

MODEL_H = input_details[0]['shape'][1]
MODEL_W = input_details[0]['shape'][2]

input_scale, input_zero_point = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']

# --- 3. DINH NGHIA LOP CAMERA DA LUONG ---
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

# =========================================================
# HAM GUI ANH CO BBOX LEN SERVER (CHAY THREAD RIENG)
def send_frame_to_server(frame_with_bbox, detections):
    """
    Nhan frame da ve bbox, encode va gui len server.
    detections: list cac dict {"label", "score", "bbox": [x,y,w,h]}
    """
    try:
        # Encode frame thanh JPEG
        success, buffer = cv2.imencode('.jpg', frame_with_bbox, 
                                        [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            return

        # Tao ten file theo timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"benh_{timestamp}.jpg"

        # Gui anh + thong tin phat hien len server
        files = {'image': (filename, buffer.tobytes(), 'image/jpeg')}
        data = {
            'timestamp': timestamp,
            'detections': str(detections)  # Gui kem thong tin bbox + label
        }

        response = requests.post(SERVER_URL, files=files, data=data, timeout=5)
        print(f"[SERVER] Gui thanh cong: {filename} | Status: {response.status_code}")

    except Exception as e:
        print(f"[SERVER] Loi gui anh: {e}")
# =========================================================

# --- 4. MO CAMERA USB ---
print("Dang khoi dong Camera USB Da luong...")
cap = ThreadedCamera(src=1).start()

x_scale = cap.raw_w / MODEL_W
y_scale = cap.raw_h / MODEL_H

frame_count = 0
fps_avg = 0
start_time = time.time()
last_results = []

print("Bat dau quet benh. Nhan 'q' de thoat.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # --- GIAI DOAN 1: SKIP FRAMES ---
    if frame_count % SKIP_FRAMES != 0:
        for box_info in last_results:
            x, y, w, h, label_name, score_percent = box_info
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_name} {score_percent}%", (x, max(10, y-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.putText(frame, f"FPS: {fps_avg:.1f} (Raw Cam)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Quet Benh Tren La (YOLOv8 MT)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # --- GIAI DOAN 2: XU LY AI ---
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

    predictions = np.squeeze(output_data).T

    boxes_list = []
    confidences = []
    class_ids = []

    for row in predictions:
        classes_scores = row[4:]
        class_id = np.argmax(classes_scores)
        score = classes_scores[class_id]

        if score > CONF_THRESHOLD:
            xc, yc, w, h = row[0], row[1], row[2], row[3]
            x_min = xc - (w / 2)
            y_min = yc - (h / 2)

            left = int(x_min * x_scale)
            top = int(y_min * y_scale)
            width = int(w * x_scale)
            height = int(h * y_scale)

            left = max(0, left)
            top = max(0, top)

            boxes_list.append([left, top, width, height])
            confidences.append(float(score))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes_list, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    last_results.clear()

    # =========================================================
    # VE BBOX VA CHUP ANH GUI SERVER
    if len(indices) > 0:
        detections_info = []  # Luu thong tin de gui kem len server

        for i in indices.flatten():
            x, y, w, h = boxes_list[i]
            label_id = class_ids[i]
            score_percent = int(confidences[i] * 100)

            if 0 <= label_id < len(labels):
                label_name = labels[label_id]
            else:
                label_name = f"Class_{label_id}"

            # Ve bbox len frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_name} {score_percent}%", (x, max(10, y-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            last_results.append([x, y, w, h, label_name, score_percent])

            # Luu thong tin detection de gui kem
            detections_info.append({
                "label": label_name,
                "score": score_percent,
                "bbox": [x, y, w, h]
            })

        # Chi gui server neu qua thoi gian cooldown
        current_time = time.time()
        if current_time - last_sent_time >= SEND_COOLDOWN:
            last_sent_time = current_time
            # Chup lai frame hien tai (da co bbox) va gui thread rieng
            frame_to_send = frame.copy()
            t = threading.Thread(
                target=send_frame_to_server,
                args=(frame_to_send, detections_info),
                daemon=True
            )
            t.start()
    # =========================================================

    inf_time = (time.time() - t_inf_start) * 1000
    fps_avg = 1 / (time.time() - start_time) * SKIP_FRAMES
    start_time = time.time()

    cv2.putText(frame, f"FPS: {fps_avg:.1f} (AI Update) | AI: {inf_time:.0f}ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Quet Benh Tren La (YOLOv8 MT)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
cv2.destroyAllWindows()
print("Da thoat chuong trinh.")