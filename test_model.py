import cv2
import numpy as np
import time
from tflite_runtime.interpreter import Interpreter

# --- 1. CẤU HÌNH THÔNG SỐ ---
MODEL_PATH = "model_benh_la_sieunhe.tflite"
LABEL_PATH = "benh_la.txt"
IMAGE_PATH = "./test/1.jpg"      # <--- Đổi tên ảnh của cậu ở đây
OUTPUT_PATH = "ket_qua_test.jpg"

CONF_THRESHOLD = 0.5            # Ngưỡng tin cậy tối thiểu
NMS_THRESHOLD = 0.4             # Ngưỡng lọc chồng chéo IoU

# --- 2. TẢI NHÃN VÀ MÔ HÌNH ---
with open(LABEL_PATH, 'r', encoding='utf-8') as f:
    labels = [line.strip() for line in f.readlines()]

print("Đang nạp mô hình TFLite...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

MODEL_H = input_details[0]['shape'][1]
MODEL_W = input_details[0]['shape'][2]
IS_FLOAT = input_details[0]['dtype'] == np.float32

# --- 3. ĐỌC VÀ TIỀN XỬ LÝ ẢNH ---
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError(f"Không tìm thấy ảnh tại: {IMAGE_PATH}")

original_h, original_w, _ = img.shape
print(f"Kích thước ảnh gốc: {original_w}x{original_h} | Mạng yêu cầu: {MODEL_W}x{MODEL_H}")

# Resize và chuyển hệ màu BGR (OpenCV) -> RGB
img_resized = cv2.resize(img, (MODEL_W, MODEL_H))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

if IS_FLOAT:
    input_data = (np.float32(img_rgb) / 127.5) - 1.0 # Chuẩn hoá [-1, 1]
else:
    input_data = img_rgb # Giữ nguyên INT8/UINT8

input_data = np.expand_dims(input_data, axis=0)

# --- 4. CHẠY SUY LUẬN ---
start_time = time.time()
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
print(f"Thời gian AI xử lý (Inference): {(time.time() - start_time) * 1000:.1f} ms")

# --- 5. HẬU XỬ LÝ & BÓC TÁCH TENSOR ---
boxes, classes, scores = None, None, None
arrays_1d = []

for det in output_details:
    raw_tensor = interpreter.get_tensor(det['index'])
    tensor = np.squeeze(raw_tensor)
    
    # Giải lượng tử hoá (Dequantize) nếu mô hình là INT8
    scale, zero_point = det['quantization']
    if scale > 0.0 and raw_tensor.dtype != np.float32:
        tensor = (tensor.astype(np.float32) - zero_point) * scale

    if len(tensor.shape) == 2 and tensor.shape[1] == 4:
        boxes = tensor
    elif len(tensor.shape) == 1 and tensor.size > 1:
        arrays_1d.append(tensor)

# Ghép đúng mảng scores và classes
if len(arrays_1d) >= 2:
    arr1, arr2 = arrays_1d[0], arrays_1d[1]
    if np.max(arr1) > 1.0 or np.all(arr1 == np.floor(arr1)):
        classes, scores = arr1, arr2
    else:
        classes, scores = arr2, arr1

# --- 6. LỌC NMS VÀ VẼ KẾT QUẢ ---
boxes_list = []
confidences = []
class_ids = []

if scores is not None:
    for i in range(len(scores)):
        if scores[i] > CONF_THRESHOLD:
            ymin, xmin, ymax, xmax = boxes[i]
            
            left = int(xmin * original_w)
            top = int(ymin * original_h)
            right = int(xmax * original_w)
            bottom = int(ymax * original_h)
            
            boxes_list.append([left, top, right - left, bottom - top])
            confidences.append(float(scores[i]))
            class_ids.append(int(classes[i]))

    # Áp dụng NMS triệt tiêu box trùng lặp
    indices = cv2.dnn.NMSBoxes(boxes_list, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    if len(indices) > 0:
        print(f"Phát hiện {len(indices)} vùng bị bệnh sau khi lọc NMS.")
        for i in indices.flatten():
            x, y, w, h = boxes_list[i]
            label_id = class_ids[i]
            score_percent = int(confidences[i] * 100)
            
            # Xử lý tên nhãn an toàn
            if 0 <= label_id < len(labels):
                label_name = labels[label_id]
            else:
                label_name = f"Class_{label_id}"
                
            # Vẽ Box và Text
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"{label_name} {score_percent}%", (x, max(10, y-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        print("Không phát hiện bệnh nào vượt qua ngưỡng độ tin cậy.")
else:
    print("Lỗi: Không đọc được output tensor từ mô hình.")

# --- 7. HIỂN THỊ VÀ LƯU ---
cv2.imwrite(OUTPUT_PATH, img)
print(f"Đã lưu ảnh kết quả tại: {OUTPUT_PATH}")

# Hiển thị ảnh (Nhấn phím bất kỳ để