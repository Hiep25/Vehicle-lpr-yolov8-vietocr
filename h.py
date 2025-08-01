import cv2
from PIL import Image
from ultralytics import YOLO
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
import os

# Tải mô hình YOLO và VietOCR
model = YOLO(r"C:\Users\dell\Downloads\CV\best.pt")  # Đường dẫn đến mô hình YOLO đã huấn luyện nhận diện biển số
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = r'C:\Users\dell\Downloads\CV\vietocr.pt'  # Đường dẫn mô hình VietOCR
config['device'] = 'cpu'          # Nếu dùng GPU: 'cuda'
vietocr_model = Predictor(config)

# Đường dẫn đến video
video_path = r"C:\Users\dell\Downloads\CV\test.mp4"
cap = cv2.VideoCapture(video_path)

print("Nhấn phím SPACE để nhận diện biển số. Nhấn ESC để thoát.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = cv2.resize(frame, (800, 450))
    cv2.imshow("Video", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                license_plate = image_rgb[y1:y2, x1:x2]
                license_plate_pil = Image.fromarray(license_plate)
                text = vietocr_model.predict(license_plate_pil)
                print("🚘 Biển số xe:", text)
                # Hiển thị ảnh biển số
                cv2.imshow("License Plate", cv2.cvtColor(license_plate, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)
                cv2.destroyWindow("License Plate")

cap.release()
try:
    cv2.destroyAllWindows()
except:
    print("⚠️ cv2.destroyAllWindows() không hoạt động trong môi trường này.")

