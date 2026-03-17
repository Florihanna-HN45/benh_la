import cv2
import numpy as np
import time
import os
from tflite_runtime.interpreter import Interpreter
import threading

# --- 1. CAU HINH HE THONG ---
MODEL_PATH = "yolov8n_benh_la_int8.tflite"
LABEL_PATH = "benh_la2.txt"

# Doc tu dong file benh_la.txt
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

# Lay thong so luong tu hoa (Quantization) de giai ma INT8
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

# --- 4. MO CAMERA USB ---
print("Dang khoi dong Camera USB Da luong...")
cap = ThreadedCamera(src=1).start() # Nho check lai cong USB (0 hoac 1)

x_scale = cap.raw_w / MODEL_W
y_scale = cap.raw_h / MODEL_H

frame_count = 0
fps_avg = 0
start_time = time.time()
last_results = []

print("Bat dau quet benh. Nhan 'q' de thoat.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame_count += 1

    # --- GIAI DOAN 1: SKIP FRAMES ---
    if frame_count % SKIP_FRAMES != 0:
        for box_info in last_results:
            x, y, w, h, label_name, score_percent = box_info
            # Ve Box cuc net bao quanh vat the
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_name} {score_percent}%", (x, max(10, y-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.putText(frame, f"FPS: {fps_avg:.1f} (Raw Cam)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Quet Benh Tren La (YOLOv8 MT)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue 

    # --- GIAI DOAN 2: XU LY AI ---
    t_inf_start = time.time()

    img_resized = cv2.resize(frame, (MODEL_W, MODEL_H))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # SUA LOI 1: Luong tu hoa dau vao (Chuan bi du lieu cho INT8)
    if input_details[0]['dtype'] == np.int8:
        # Neu mo hinh la INT8 nguyen thuy, phai nen anh vao thang so nguyen
        input_data = (np.float32(img_rgb) / 255.0) / input_scale + input_zero_point
        input_data = np.clip(input_data, -128, 127).astype(np.int8)
    else:
        # Neu la mo hinh Float32
        input_data = np.float32(img_rgb) / 255.0
        
    input_data = np.expand_dims(input_data, axis=0)

    # Chay AI
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # SUA LOI 2: Giai luong tu hoa dau ra (Keo Bbox ve dung toa do that)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    if output_details[0]['dtype'] == np.int8 and output_scale > 0:
        # Giai ma ma tran tu -128->127 ve toa do pixel thuc te
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

            # Bao ve de tranh toa do bi ve ra ngoai vien man hinh
            left = max(0, left)
            top = max(0, top)

            boxes_list.append([left, top, width, height])
            confidences.append(float(score))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes_list, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    last_results.clear() 

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes_list[i]
            label_id = class_ids[i]
            score_percent = int(confidences[i] * 100)
            
            if 0 <= label_id < len(labels):
                label_name = labels[label_id]
            else:
                label_name = f"Class_{label_id}"
                
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_name} {score_percent}%", (x, max(10, y-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            last_results.append([x, y, w, h, label_name, score_percent])

    inf_time = (time.time() - t_inf_start) * 1000
    fps_avg = 1 / (time.time() - start_time) * SKIP_FRAMES
    start_time = time.time()

    cv2.putText(frame, f"FPS: {fps_avg:.1f} (AI Update) | AI: {inf_time:.0f}ms", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Quet Benh Tren La (YOLOv8 MT)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.stop()
cv2.destroyAllWindows()
print("Da thoat chuong trinh.")