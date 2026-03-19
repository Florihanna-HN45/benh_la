import cv2
import numpy as np
import tensorflow as tf
import time

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="yolov8n_benh_la_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_size = input_details[0]['shape'][1]  # lấy kích thước từ model (vd: 320 hoặc 640)

# Preprocess image
def preprocess(img, input_size):
    h, w, _ = img.shape
    img_resized = cv2.resize(img, (input_size, input_size))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0).astype(np.float32)
    return img_resized, (h, w)

# Convert xywh → xyxy
def xywh2xyxy(x):
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # xmin
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # ymin
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # xmax
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # ymax
    return y

# Non-Max Suppression
def nms(boxes, scores, iou_threshold=0.1):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        ious = compute_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_threshold]
    return keep

def compute_iou(box, boxes):
    inter_xmin = np.maximum(box[0], boxes[:, 0])
    inter_ymin = np.maximum(box[1], boxes[:, 1])
    inter_xmax = np.minimum(box[2], boxes[:, 2])
    inter_ymax = np.minimum(box[3], boxes[:, 3])

    inter_area = np.maximum(0, inter_xmax - inter_xmin) * np.maximum(0, inter_ymax - inter_ymin)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area
    return inter_area / union_area

# Camera realtime
url = "http://192.168.2.74:8080/video"
cap = cv2.VideoCapture(url)  # 0 = webcam mặc định
frame_skip = 3  # bỏ qua 2-3 frames, chỉ xử lý 1 frame
frame_count = 0
fps = 0
prev_time = time.time()

while True:
    ret, img = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        # bỏ qua frame này
        continue

    # Preprocess
    input_data, (orig_h, orig_w) = preprocess(img, input_size)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Decode output
    boxes = output_data[:, :4]
    scores = output_data[:, 4]
    classes = np.argmax(output_data[:, 5:], axis=1)

    boxes = xywh2xyxy(boxes)
# Nếu output đã normalized (0–1), scale theo ảnh gốc:
    boxes[:, [0, 2]] *= orig_w
    boxes[:, [1, 3]] *= orig_h

    keep = nms(boxes, scores)
    for i in keep:
        if scores[i] > 0.5:
            xmin, ymin, xmax, ymax = boxes[i].astype(int)
            cls_id = classes[i]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img, f"Class {cls_id} {scores[i]:.2f}",
                        (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    # FPS tính toán
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(img, f"FPS: {fps:.2f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("YOLOv8 TFLite Realtime", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()