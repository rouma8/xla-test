# 🚗 Nhận dạng Biển số xe Việt Nam

Ứng dụng nhận dạng biển số xe Việt Nam sử dụng các kỹ thuật xử lý ảnh cổ điển.

## Công nghệ sử dụng
- **Python** + **OpenCV** 
- Phát hiện biên: **Canny Edge Detection**
- Xử lý hình thái: **Dilation, Closing**
- Phân ngưỡng: **Otsu Thresholding**
- Nhận dạng ký tự: **KNN (K-Nearest Neighbors)**

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng

### Chạy với giao diện (GUI)
```bash
python app.py
```

### Chạy dòng lệnh
```bash
python main.py -i dataset/plates/xe1.jpg
```

### Tham số
| Tham số | Mô tả |
|---------|--------|
| `-i`, `--image` | Đường dẫn ảnh đầu vào (bắt buộc) |
| `-d`, `--dataset` | Đường dẫn thư mục dataset ký tự (mặc định: `dataset/chars`) |
| `--no-show` | Không hiển thị ảnh kết quả |

## Cấu trúc thư mục

```
p_xla/
├── dataset/
│   ├── plates/              # Ảnh xe để test
│   └── chars/               # Dataset ký tự (train KNN)
│       ├── 0/ ... 9/        # Thư mục ảnh số
│       └── A/ ... Z/        # Thư mục ảnh chữ
├── src/
│   ├── preprocess.py        # Tiền xử lý ảnh
│   ├── plate_detector.py    # Phát hiện biển số
│   ├── char_segmentation.py # Phân đoạn ký tự
│   └── char_recognition.py  # Nhận dạng ký tự (KNN)
├── app.py                   # Giao diện Tkinter
├── main.py                  # Pipeline dòng lệnh
├── requirements.txt
└── README.md
```

## Pipeline xử lý

```
Ảnh đầu vào → Grayscale → Lọc nhiễu (Bilateral)
→ Cân bằng Histogram (CLAHE) → Phát hiện biên (Canny)
→ Xử lý hình thái (Closing + Dilation) → Tìm Contour
→ Cắt biển số → Phân ngưỡng (Otsu) → Tách ký tự
→ Nhận dạng KNN → Kết quả biển số
```

## Thành viên nhóm
- Thành viên A: Tiền xử lý & Phát hiện biển số
- Thành viên B: Phân đoạn & Nhận dạng ký tự
- Thành viên C: Tích hợp, Giao diện & Báo cáo
