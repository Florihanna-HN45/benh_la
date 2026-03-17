# Save file as: yolo_pi_multithread.py
import cv2
import numpy as np
import time
from tflite_runtime.interpreter import Interpreter
import threading

# --- 1. CẤU HÌNH HỆ THỐNG ---
# Tải tệp mô hình best_int8.tflite của cậu về Pi
MODEL_PATH = "yolov8n_benh_la_int8.tflite"
# Mở file benh_la.txt trên Pi để check tên nhãn có khớp không
labels = ["Dom_La", "Heo_Ranh", "Khoe_Manh"] # Sửa lại theo model của cậu

CONF_THRESHOLD = 0.5            # Ngưỡng tin cậy
NMS_THRESHOLD = 0.4             # Ngưỡng lọc trùng IoU

# Kỹ thuật Skip Frames: Cứ 4 frame camera, xử lý AI cho 1 frame
SKIP_FRAMES = 4

# --- 2. NẠP MÔ HÌNH TFLITE ---
print("Đang nạp mô hình YOLOv8 TFLite...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Mạng YOLOv8 yêu cầu hình vuông (ví dụ 320x320)
MODEL_H = input_details[0]['shape'][1]
MODEL_W = input_details[0]['shape'][2]

# --- 3. ĐỊNH NGHĨA LỚP CAMERA ĐA LUỒNG (THREADED CAMERA) ---
class ThreadedCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.raw_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        # Kích hoạt luồng đọc camera riêng biệt
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Vòng lặp liên tục đọc frame camera
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        # Lấy frame mới nhất
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# --- 4. MỞ CAMERA USB ĐA LUỒNG ---
print("Đang khởi động Camera USB Đa luồng...")
cap = ThreadedCamera(src=0).start() # Camera USB mặc định

# Tính trước tỷ lệ Scale tọa độ từ mô hình về kích thước ảnh thật
x_scale = cap.raw_w / MODEL_W
y_scale = cap.raw_h / MODEL_H

frame_count = 0
fps_avg = 0
start_time = time.time()
# Mảng lưu kết quả AI của frame trước đó để vẽ bù cho các frame bị skip
last_results = []

print("Bắt đầu quét bệnh. Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame_count += 1

    # --- GIAI ĐOẠN 1: KỸ THUẬT SKIP FRAMES (drawing bù data cũ) ---
    if frame_count % SKIP_FRAMES != 0:
        # Nhờ có mảng 'last_results', các khung hình bị skip vẫn có khung bao 
        # (Hiệu ứng thị giác giúp video trông rất mượt)
        for box_info in last_results:
            x, y, w, h, label_name, score_percent = box_info
            # Vẽ Box (màu XANH LÁ: (0, 255, 0))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Vẽ Text (màu XANH DƯƠNG: (255, 0, 0))
            cv2.putText(frame, f"{label_name} {score_percent}%", (x, max(10, y-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Hiển thị FPS và trạng thái Raw Camera (màu xanh lá)
        cv2.putText(frame, f"FPS: {fps_avg:.1f} (Raw Cam)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Quet Benh Tren La (YOLOv8 MT)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue 

    # --- GIAI ĐOẠN 2: XỬ LÝ AI CHO FRAME CHÍNH ---
    t_inf_start = time.time()

    # Tiền xử lý ảnh cho YOLOv8 (Resize và hệ màu RGB)
    img_resized = cv2.resize(frame, (MODEL_W, MODEL_H))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # YOLOv8 chuẩn mô hình FP16/FP32 yêu cầu chuẩn hóa về [0, 1] (khác SSD)
    input_data = np.float32(img_rgb) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    # Nạp dữ liệu vào mô hình
    interpreter.set_tensor(input_details[0]['index'], input_data)
    # GỌI SUY LUẬN (Nặng nhất trên Raspi)
    interpreter.invoke()

    # Bóc tách ma trận kết quả duy nhất của YOLOv8: [1, 4+classes, 8400]
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # Xoay lại thành ma trận chuẩn dễ lặp: [8400, 4+classes]
    predictions = np.squeeze(output_data).T 

    boxes_list = []
    confidences = []
    class_ids = []

    for row in predictions:
        classes_scores = row[4:]
        class_id = np.argmax(classes_scores)
        score = classes_scores[class_id]

        # Kiểm tra ngưỡng độ tin cậy
        if score > CONF_THRESHOLD:
            # Tọa độ YOLOv8 là (x_tâm, y_tâm, w, h)
            xc, yc, w, h = row[0], row[1], row[2], row[3]
            
            # Chuyển đổi công thức toán học về tọa độ OpenCV (x_min, y_min)
            $x_{min} = x_c - \frac{w}{2}$
            $y_{min} = y_c - \frac{h}{2}$
            
            # Scale tọa độ pixel từ mô hình về kích thước ảnh thật
            left = int(x_min * x_scale)
            top = int(y_min * y_scale)
            width = int(w * x_scale)
            height = int(h * y_scale)

            # Thu thập dữ liệu để lọc NMS
            boxes_list.append([left, top, width, height])
            confidences.append(float(score))
            class_ids.append(class_id)

    # Chạy thuật toán lọc trùng NMS
    indices = cv2.dnn.NMSBoxes(boxes_list, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    last_results.clear() # Xóa data cũ để cập nhật data mới

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes_list[i]
            label_id = class_ids[i]
            score_percent = int(confidences[i] * 100)
            
            # Xử lý tên nhãn an toàn
            if 0 <= label_id < len(labels):
                label_name = labels[label_id]
            else:
                label_name = f"Class_{label_id}"
                
            # Vẽ Box (XANH LÁ) và Text (XANH DƯƠNG) lên frame chính
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_name} {score_percent}%", (x, max(10, y-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Lưu lại để vẽ bù cho các frame bị skip sau này
            last_results.append([x, y, w, h, label_name, score_percent])

    # Tính toán thông số thời gian
    inf_time = (time.time() - t_inf_start) * 1000
    fps_avg = 1 / (time.time() - start_time) * SKIP_FRAMES
    start_time = time.time()

    # Hiển thị FPS và trạng thái AI (màu đỏ)
    cv2.putText(frame, f"FPS: {fps_avg:.1f} (AI Update) | AI: {inf_time:.0f}ms", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Quet Benh Tren La (YOLOv8 MT)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.stop() # Bắt buộc phải stop đa luồng trước khi thoát
cv2.destroyAllWindows()
print("Đã thoát chương trình.")