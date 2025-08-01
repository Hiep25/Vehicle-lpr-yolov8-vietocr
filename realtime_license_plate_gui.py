import cv2
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import torch
import numpy as np
from datetime import datetime
import os

# Load YOLOv8 model
yolo_model = YOLO(r"E:\ComputerVision\FinalReport\best.pt")  # Đảm bảo file best.pt nằm cùng thư mục 

# Load VietOCR model
config = Cfg.load_config_from_name('vgg_transformer')
config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
ocr_model = Predictor(config)

def detect_plate(frame):
    # Nhận diện và đọc biển số từ frame.
    results = yolo_model(frame)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        if len(boxes) == 0:
            return "Không tìm thấy", None

        x1, y1, x2, y2 = boxes[0]
        cropped_plate = frame[y1:y2, x1:x2]
        if cropped_plate.shape[0] < 10 or cropped_plate.shape[1] < 10:
            return "Biển số quá nhỏ", None

        plate_pil = Image.fromarray(cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB))

        try:
            plate_text = ocr_model.predict(plate_pil)
        except:
            plate_text = "Lỗi OCR"

        return plate_text.strip(), cropped_plate
    return "Không phát hiện", None

# Tạo thư mục lưu ảnh nếu chưa có
os.makedirs("captures", exist_ok=True)

# GUI chính
cap = cv2.VideoCapture(0)
saved_results = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    plate_text, plate_crop = detect_plate(display_frame)

    # Hiển thị biển số lên hình ảnh
    cv2.putText(display_frame, f"Bien so: {plate_text}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hiển thị kết quả đã lưu
    y_offset = 80
    for i, (img, text) in enumerate(saved_results[-3:]):
        small = cv2.resize(img, (200, 60))
        display_frame[y_offset:y_offset+60, -220:-20] = small
        cv2.putText(display_frame, text, (-220 + 10, y_offset + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 70

    cv2.imshow("License Plate Recognition", display_frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC để thoát
        break
    elif key == 32:  # Space để lưu
        if plate_crop is not None:
            filename = f"captures/plate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, plate_crop)
            saved_results.append((plate_crop, plate_text))

cap.release()
cv2.destroyAllWindows()
