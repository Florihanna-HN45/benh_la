import cv2
import numpy as np
import time
from tflite_runtime.interpreter import Interpreter

# --- CẤU HÌNH HỆ THỐNG ---
MODEL_PATH = "model_benh_la_sieunhe.tflite"
LABEL_PATH = "labels.txt"
threshold = 0.5  # Độ tin cậy tối thiểu (0.0 -> 1.0) để vẽ box

# Kỹ thuật 1: Bỏ khung hình (Skip Frames)
# Cứ 4 khung từ camera, bỏ 3, chỉ xử lý 1
# Số càng cao càng mượt nhưng AI cập nhật chậm
SKIP_FRAMES = 4 

# --- NẠP MÔ HÌNH VÀ TỪ ĐIỂN ---
with open(LABEL_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Nạp TFLite và kích hoạt bộ tăng tốc XNNPACK trên CPU ARM
# Thao tác này giúp ma trận chạy nhanh hơn mặc định
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Lấy thông tin đầu vào/đầu ra của mô hình
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Mô hình được train ở kích thước nào (Ví dụ: 320x320)
MODEL_H, MODEL_W = input_details[0]['shape'][1], input_details[0]['shape'][2]

# --- MỞ CAMERA USB ---
print("Đang khởi động Camera USB...")
cap = cv2.VideoCapture(1) # Camera USB mặc định

# Cấu hình camera lấy FPS thấp ngay từ đầu để giảm tải bus USB
cap.set(cv2.CAP_PROP_FPS, 15) 

# Lấy kích thước thật của camera
raw_cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
raw_cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_count = 0
fps_avg = 0
start_time = time.time()

print("Bắt đầu quét bệnh trên lá. Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1
    
    # ---------------------------------------------------------
    # GIAI ĐOẠN 1: KỸ THUẬT SKIP FRAMES (BỎ KHUNG HÌNH)
    # ---------------------------------------------------------
    if frame_count % SKIP_FRAMES != 0:
        # Nếu không phải khung hình cần xử lý AI, chỉ hiển thị ảnh camera
        cv2.putText(frame, f"FPS: {fps_avg:.1f} (Raw Cam)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Quet Benh Tren La (USB Cam)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue # Đi đến khung hình tiếp theo luôn

    # ---------------------------------------------------------
    # GIAI ĐOẠN 2: CHỈ KHUNG HÌNH THỨ 4, 8, 12... MỚI VÀO ĐƯỢC ĐÂY
    # ---------------------------------------------------------
    t_inf_start = time.time() # Đo thời gian xử lý AI

    # Kỹ thuật 2: Giảm độ phân giải (Downscaling)
    # Resize về kích thước mô hình cần (ví dụ 320x320)
    input_image = cv2.resize(frame, (MODEL_W, MODEL_H))
    
    # Kỹ thuật 3: Tiền xử lý dữ liệu chuẩn TFLite
    # Chuyển ảnh màu BGR của OpenCV thành RGB, và chuẩn hóa (Float32 hoặc INT8)
    if input_details[0]['dtype'] == np.float32:
        input_data = (np.float32(input_image) / 127.5) - 1.0 # Chuẩn hóa về [-1, 1]
    else:
        input_data = input_image # Dùng trực tiếp nếu là mô hình đã Quantize (INT8)
    input_data = np.expand_dims(input_data, axis=0) # Thêm trục batch (1, 320, 320, 3)

    # Đưa dữ liệu vào mô hình
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # GỌI SUY LUẬN (Nặng nhất trên Raspi)
    interpreter.invoke()

    # Lấy kết quả đầu ra
    # output_details[0] là boxes, [1] là classes, [2] là scores
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] 
    classes = interpreter.get_tensor(output_details[1]['index'])[0] 
    scores = interpreter.get_tensor(output_details[2]['index'])[0] 

    # --- HẬU XỬ LÝ: VẼ KẾT QUẢ ---
    for i in range(len(scores)):
        if scores[i] > threshold:
            # Lấy tọa độ box (dạng tỷ lệ 0-1)
            ymin, xmin, ymax, xmax = boxes[i]
            
            # Chuyển về tọa độ pixel thật của camera để vẽ
            left = int(xmin * raw_cam_w)
            top = int(ymin * raw_cam_h)
            right = int(xmax * raw_cam_w)
            bottom = int(ymax * raw_cam_h)
            
            label_id = int(classes[i])
            label_name = labels[label_id]
            score_percent = int(scores[i] * 100)
            
            # Vẽ Box và Tên bệnh lên ảnh gốc (frame)
            cv2.rectangle(frame, (left, top), (right, bottom), (10, 255, 0), 3)
            cv2.putText(frame, f"{label_name} {score_percent}%", (left, top-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Tính toán FPS thực tế
    t_inf_end = time.time()
    inf_time = (t_inf_end - t_inf_start) * 1000 # Thời gian xử lý 1 khung hình (ms)
    fps_avg = 1 / (time.time() - start_time)
    start_time = time.time()

    # Hiển thị thông số và ảnh kết quả
    cv2.putText(frame, f"FPS: {fps_avg:.1f} (AI Update) | Time: {inf_time:.0f}ms", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Quet Benh Tren La (USB Cam)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
print("Đã thoát chương trình.")