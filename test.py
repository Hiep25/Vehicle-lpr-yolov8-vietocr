import locale
locale.getpreferredencoding = lambda: "UTF-8"

# ========== Load model ==========
from vietocr.tool.config import Cfg
from vietocr.predict import Predictor
from ultralytics import YOLO

# VietOCR config
config = Cfg.load_config_from_name('vgg_transformer')
config['device'] = 'cuda'  # hoặc 'cpu' nếu không có GPU
vietocr_model = Predictor(config)

# YOLOv8
model = YOLO("E:/ComputerVision/FinalReport/best.pt")  # Đường dẫn model của bạn

# ========== Xử lý ảnh ==========
import urllib.request
import numpy as np
import cv2
from PIL import Image
import gradio as gr

def detect_and_recognize_plate(image: Image.Image):
    # Chuyển PIL -> OpenCV
    image_np = np.array(image)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Dự đoán với YOLOv8
    results = model(image_cv2)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    if len(boxes) == 0:
        return "Không phát hiện biển số"

    plate_texts = []

    for box in boxes:
        x1, y1, x2, y2 = box
        cropped_plate = image_np[y1:y2, x1:x2]
        cropped_img_pil = Image.fromarray(cropped_plate)

        try:
            text = vietocr_model.predict(cropped_img_pil)
            plate_texts.append(text)
        except Exception as e:
            plate_texts.append(f"Lỗi nhận dạng: {e}")

    return ", ".join(plate_texts)

# ========== Gradio Interface ==========
gr.Interface(
    fn=detect_and_recognize_plate,
    inputs=gr.Image(type="pil", source="upload", label="Ảnh biển số hoặc webcam"), kk
    outputs="text",
    title="Nhận diện biển số xe",
    description="Sử dụng YOLOv8 để phát hiện và VietOCR để nhận diện ký tự.",
    live=False
).launch()
