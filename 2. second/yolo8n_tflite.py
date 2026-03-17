import cv2
import numpy as np
import time
from tflite_runtime.interpreter import Interpreter

# --- 1. CẤU HÌNH HỆ THỐNG ---
MODEL_PATH = "yolov8n_benh_la_int8.tflite"
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
labels = ["Fully Nutritional", "Natrium","Phosphorus", "Potassium"]

# Kỹ thuật Skip Frames: Cứ 4 frame thì xử lý AI 1 frame
SKIP_FRAMES = 4 

# --- 2. NẠP MÔ HÌNH TFLITE ---
print("Đang nạp mô hình YOLOv8 TFLite...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

MODEL_W = input_details[0]['shape'][2]
MODEL_H = input_details[0]['shape'][1]
IS_FLOAT = input_details[0]['dtype'] == np.float32

# --- 3. MỞ CAMERA USB ---
print("Đang khởi động Camera USB...")
cap = cv2.VideoCapture(0) # Đổi thành 1 nếu Pi nhận USB cam là video1
cap.set(cv2.CAP_PROP_FPS, 30)

raw_cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
raw_cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Tính trước tỷ lệ Scale để tối ưu tốc độ trong vòng lặp
x_scale = raw_cam_w / MODEL_W
y_scale = raw_cam_h / MODEL_H

frame_count = 0
fps_avg = 0
start_time = time.time()
last_results = [] # Lưu trữ kết quả AI của frame trước đó để vẽ bù

print("Bắt đầu quét bệnh. Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1
    
    # --- GIAI ĐOẠN 1: KỸ THUẬT SKIP FRAMES ---
    if frame_count % SKIP_FRAMES != 0:
        # Nhờ có biến last_results, các frame bị bỏ qua AI vẫn có khung xanh 
        # (Hiệu ứng thị giác giúp video trông rất mượt)
        for box_info in last_results:
            x, y, w, h, label_name, score_percent = box_info
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_name} {score_percent}%", (x, max(10, y-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        cv2.putText(frame, f"FPS: {fps_avg:.1f} (Skip)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Quet Benh Tren La (YOLOv8)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue 

    # --- GIAI ĐOẠN 2: XỬ LÝ AI CHO FRAME CHÍNH ---
    t_inf_start = time.time()

    # Tiền xử lý ảnh cho YOLOv8
    img_resized = cv2.resize(frame, (MODEL_W, MODEL_H))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    if IS_FLOAT:
        input_data = np.float32(img_rgb) / 255.0
    else:
        input_data = img_rgb
        
    input_data = np.expand_dims(input_data, axis=0)

    # Chạy suy luận
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Bóc tách ma trận YOLOv8
    output_data = interpreter.get_tensor(output_details[0]['index'])
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
            
            # Đổi từ tâm sang tọa độ góc và Scale lên kích thước thật
            left = int((xc - w/2) * x_scale)
            top = int((yc - h/2) * y_scale)
            width = int(w * x_scale)
            height = int(h * y_scale)

            boxes_list.append([left, top, width, height])
            confidences.append(float(score))
            class_ids.append(class_id)

    # Lọc NMS
    indices = cv2.dnn.NMSBoxes(boxes_list, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    last_results.clear() # Xóa data cũ để cập nhật data mới

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes_list[i]
            label_id = class_ids[i]
            score_percent = int(confidences[i] * 100)
            label_name = labels[label_id] if 0 <= label_id < len(labels) else f"Class_{label_id}"
            
            # Lưu lại để vẽ cho các frame bị skip phía sau
            last_results.append([x, y, w, h, label_name, score_percent])
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_name} {score_percent}%", (x, max(10, y-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Tính toán thông số thời gian
    inf_time = (time.time() - t_inf_start) * 1000
    fps_avg = 1 / (time.time() - start_time) * SKIP_FRAMES
    start_time = time.time()

    cv2.putText(frame, f"FPS: {fps_avg:.1f} | AI: {inf_time:.0f}ms", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Quet Benh Tren La (YOLOv8)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
print("Đã thoát chương trình.")