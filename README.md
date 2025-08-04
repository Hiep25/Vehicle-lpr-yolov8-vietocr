🚘 Hệ Thống Nhận Diện Biển Số Xe bằng YOLOv8 và VietOCR

Dự án xây dựng một hệ thống nhận diện biển số xe máy, xe ô tô tự động bằng cách kết hợp mô hình phát hiện đối tượng **YOLOv8** với mô hình nhận dạng ký tự tiếng Việt **VietOCR**. Hệ thống hướng đến ứng dụng thực tế trong quản lý bãi đỗ xe, giúp giảm thiểu thao tác thủ công, tăng độ chính xác và hiệu quả vận hành.

<img width="831" height="381" alt="image" src="https://github.com/user-attachments/assets/5cb35457-2340-4b2c-8074-79cf71a6079f" />
<img width="482" height="504" alt="image" src="https://github.com/user-attachments/assets/12a32549-5c8a-4591-94d6-a22ebc46d0cc" />
<img width="482" height="504" alt="image" src="https://github.com/user-attachments/assets/ff69864f-e10d-4a76-a335-524b9af02c1d" />




---

 🎯 Mục Tiêu Dự Án

- Phát hiện và trích xuất vùng chứa biển số xe máy trong ảnh sử dụng YOLOv8n.
- Nhận dạng chuỗi ký tự biển số bằng mô hình VietOCR (Transformer + CTC).
- Xây dựng pipeline đầy đủ: đầu vào ảnh → phát hiện → crop → nhận dạng → xuất kết quả.
- Đánh giá độ chính xác hệ thống và đề xuất hướng cải tiến.

---

 🧱 Kiến Trúc Hệ Thống

```

Ảnh đầu vào → \[YOLOv8n] → Bounding Box biển số → Cắt ảnh → \[VietOCR] → Chuỗi ký tự

````

 Thành phần chính:
1. **YOLOv8n** – Phát hiện đối tượng (biển số)
2. **Xử lý ảnh** – Cắt ảnh theo bounding box, resize chuẩn đầu vào VietOCR
3. **VietOCR** – Nhận dạng ký tự bằng VGG-Transformer
4. **Giao diện hiển thị (tùy chọn)** – Xuất kết quả văn bản và ảnh đầu ra

---

 🛠️ Công Nghệ & Mô Hình

| Thành phần         | Công nghệ/Framework                        |
|--------------------|--------------------------------------------|
| Ngôn ngữ           | Python 3.10                                |
| Phát hiện biển số  | YOLOv8n (Ultralytics, pretrained COCO)     |
| OCR ký tự          | VietOCR (VGG + Transformer + CTC)          |
| Xử lý ảnh          | OpenCV, NumPy                              |
| Môi trường chạy    | Google Colab (GPU), Jupyter Notebook       |
| Trực quan hóa      | Matplotlib, Pandas                         |
| Quản lý mã nguồn   | Git, GitHub                                |

---

 📂 Cấu Trúc Dự Án (đề xuất)

```bash
.
├── yolov8/                  # Training và inference YOLOv8
│   └── detect.py
├── vietocr/                 # Nhận dạng chuỗi ký tự từ ảnh
│   └── recognize.py
├── data/                    # Ảnh đã gán nhãn (train/val/test)
├── results/                 # Ảnh đầu ra, ảnh crop và chuỗi kết quả
├── assets/                  # Ảnh minh họa cho README
├── requirements.txt         # Thư viện cần cài
└── README.md                # Mô tả dự án (file này)
````

---

 🧪 Kết Quả Thực Nghiệm

 📍 YOLOv8n – Phát hiện biển số

* **Precision**: 91.2%
* **Recall**: 87.7%
* **mAP\@0.5**: 90.3%
* **mAP\@0.5:0.95**: 61.0%
* **Thời gian suy luận**: \~18–25ms/ảnh

 📍 VietOCR – Nhận dạng ký tự

* **Character Accuracy (CA)**: \~95.2%
* **Sequence Accuracy (SA)**: \~89.4%
* **Ví dụ đúng**: `59C2-345.67`
* **Ví dụ sai nhẹ**: `30F9-988.38` → `30F9-98B.38`

---

 📈 Minh Họa Kết Quả

 1. Ảnh gốc đầu vào
![Input]<img width="831" height="381" alt="image" src="https://github.com/user-attachments/assets/5cb35457-2340-4b2c-8074-79cf71a6079f" />

 2. Phát hiện biển số bằng YOLOv8
![YOLO Detection]<img width="620" height="324" alt="image" src="https://github.com/user-attachments/assets/2aaad8f8-de44-4b2c-a6f9-f361893de448" />

 3. Vùng biển số đã crop
![Cropped Plate]<img width="330" height="115" alt="image" src="https://github.com/user-attachments/assets/e2830b66-b6bb-4e5b-be73-7a4568c8c442" />

 4. Kết quả nhận dạng bằng VietOCR
![OCR Result]<img width="633" height="308" alt="image" src="https://github.com/user-attachments/assets/83b27a94-0366-42f5-a17f-880ce50f72fd" />

```

 🧠 Ưu Điểm

* Pipeline chạy ổn định, dễ triển khai.
* Thời gian xử lý nhanh (\~20ms), phù hợp thời gian thực.
* Độ chính xác cao với dữ liệu ảnh rõ nét.
* Có thể tích hợp vào hệ thống camera giám sát bãi xe.

---

 ⚠️ Hạn Chế

* Biển số bị che, nghiêng hoặc ảnh mờ làm giảm độ chính xác.
* Không nhận dạng được trong điều kiện ánh sáng quá yếu hoặc quá sáng.
* Cần fine-tune thêm VietOCR với dữ liệu thực tế để đạt kết quả tốt hơn.

---

 🔄 Định Hướng Phát Triển

* Triển khai mô hình thực tế trên thiết bị nhúng (Jetson Nano, Raspberry Pi).
* Phát triển giao diện web giám sát, lưu log biển số vào cơ sở dữ liệu.
* Bổ sung cơ chế hậu xử lý chuỗi để kiểm tra định dạng và giảm lỗi chính tả.
* Tăng cường dữ liệu với augmentation: làm mờ, xoay, thay đổi độ sáng, noise.

---

 ▶️ Hướng Dẫn Sử Dụng (nếu tự chạy)

 1. Cài thư viện
pip install -r requirements.txt

 2. Phát hiện biển số
python yolov8/detect.py --source ./data/test/sample.jpg

 3. Nhận dạng ký tự từ ảnh đã crop
python vietocr/recognize.py --image ./results/cropped_plate.jpg

 Có thể thay bằng notebook Colab nếu bạn không chạy local.

 🤝 Liên hệ

Bạn có thể liên hệ với mình nếu có thắc mắc hoặc muốn góp ý cho dự án:

**Email:** [hiepbt17@gmail.com]
