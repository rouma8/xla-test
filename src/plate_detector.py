"""
Module phát hiện biển số xe (Plate Detection)
- Phát hiện biên bằng Canny
- Xử lý hình thái (morphology)
- Tìm contour hình chữ nhật
- Cắt vùng biển số

Áp dụng kiến thức:
  Chương 5: Phát hiện biên (Canny)
  Chương 7: Xử lý hình thái (Dilation, Closing)
"""

import cv2
import numpy as np


def detect_edges(image, low_threshold=30, high_threshold=200):
    """Phát hiện biên bằng thuật toán Canny.

    Áp dụng: Chương 5 - Phát hiện biên tối ưu (Canny)
    Canny gồm các bước:
        1. Lọc Gaussian làm mịn
        2. Tính gradient (đạo hàm bậc 1)
        3. Triệt tiêu phi cực đại (non-maximum suppression)
        4. Ngưỡng kép (double threshold) + theo dõi biên (edge tracking)

    Args:
        image: Ảnh grayscale đã tiền xử lý
        low_threshold: Ngưỡng thấp
        high_threshold: Ngưỡng cao
    """
    return cv2.Canny(image, low_threshold, high_threshold)


def apply_morphology(edge_image, kernel_size=(3, 3), iterations=1):
    """Xử lý hình thái để nối các biên bị đứt đoạn.

    Áp dụng: Chương 7 - Xử lý hình thái
    - Dilation: Mở rộng vùng trắng, nối các biên gần nhau
    - Closing: Dilation rồi Erosion, lấp đầy lỗ nhỏ

    Args:
        edge_image: Ảnh biên (binary) từ Canny
        kernel_size: Kích thước phần tử cấu trúc (SE)
        iterations: Số lần lặp phép toán
    """
    # Tạo phần tử cấu trúc (Structuring Element) hình chữ nhật
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # Phép đóng (closing = dilation + erosion): lấp khoảng trống giữa các biên
    closed = cv2.morphologyEx(edge_image, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    # Phép giãn (dilation): làm dày biên để dễ phát hiện contour
    dilated = cv2.dilate(closed, kernel, iterations=iterations)

    return dilated


def find_plate_contours(morph_image, original_image):
    """Tìm contour hình chữ nhật có tỷ lệ giống biển số xe.

    Áp dụng: Chương 5 - Phân vùng bằng phát hiện biên

    Biển số xe Việt Nam có tỷ lệ:
    - 1 hàng (dài): width/height ≈ 3.5 - 5.0
    - 2 hàng (vuông): width/height ≈ 1.0 - 2.0

    Args:
        morph_image: Ảnh sau xử lý hình thái
        original_image: Ảnh gốc (để lấy kích thước tham chiếu)

    Returns:
        Danh sách các contour tiềm năng là biển số, đã sắp xếp theo diện tích giảm dần
    """
    contours, _ = cv2.findContours(morph_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sắp xếp theo diện tích giảm dần, lấy top 10
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    img_h, img_w = original_image.shape[:2]
    img_area = img_h * img_w
    plate_candidates = []

    for contour in contours:
        # Xấp xỉ đa giác
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Biển số là hình chữ nhật (4 đỉnh)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            area_ratio = area / img_area

            # Lọc theo tỷ lệ khung hình và diện tích
            is_long_plate = 2.5 <= aspect_ratio <= 6.0     # Biển 1 hàng
            is_square_plate = 0.8 <= aspect_ratio <= 2.2   # Biển 2 hàng
            is_valid_size = 0.005 <= area_ratio <= 0.3      # Kích thước hợp lý

            if (is_long_plate or is_square_plate) and is_valid_size:
                plate_candidates.append({
                    "contour": approx,
                    "bbox": (x, y, w, h),
                    "aspect_ratio": aspect_ratio,
                    "area": area,
                    "type": "1_hang" if is_long_plate else "2_hang"
                })

    # Sắp xếp theo diện tích giảm dần
    plate_candidates.sort(key=lambda c: c["area"], reverse=True)
    return plate_candidates


def crop_plate(image, bbox, padding=5):
    """Cắt vùng biển số từ ảnh gốc.

    Args:
        image: Ảnh gốc
        bbox: Bounding box (x, y, w, h)
        padding: Số pixel mở rộng xung quanh

    Returns:
        Ảnh biển số đã cắt
    """
    x, y, w, h = bbox
    img_h, img_w = image.shape[:2]

    # Thêm padding, đảm bảo không vượt biên ảnh
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)

    return image[y1:y2, x1:x2]


def detect_plate(processed_image, original_image):
    """Pipeline phát hiện biển số hoàn chỉnh.

    Quy trình:
        1. Phát hiện biên Canny (Chương 5)
        2. Xử lý hình thái - nối biên (Chương 7)
        3. Tìm contour hình chữ nhật
        4. Cắt vùng biển số

    Args:
        processed_image: Ảnh grayscale đã tiền xử lý
        original_image: Ảnh gốc (BGR)

    Returns:
        tuple: (ảnh biển số đã cắt, thông tin biển số, ảnh gốc có đánh dấu)
               Trả về (None, None, original_image) nếu không tìm thấy
    """
    # Bước 1: Phát hiện biên
    edges = detect_edges(processed_image)

    # Bước 2: Xử lý hình thái
    morph = apply_morphology(edges, kernel_size=(5, 5), iterations=2)

    # Bước 3: Tìm contour biển số
    candidates = find_plate_contours(morph, original_image)

    if not candidates:
        return None, None, original_image

    # Lấy candidate tốt nhất (diện tích lớn nhất)
    best = candidates[0]

    # Bước 4: Cắt vùng biển số
    plate_img = crop_plate(original_image, best["bbox"])

    # Vẽ bounding box lên ảnh gốc để hiển thị
    marked_image = original_image.copy()
    x, y, w, h = best["bbox"]
    cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(marked_image, f'Bien so ({best["type"]})',
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return plate_img, best, marked_image


def get_detection_steps(processed_image, original_image):
    """Trả về ảnh ở từng bước phát hiện (dùng để hiển thị minh họa).

    Returns:
        dict chứa ảnh ở mỗi bước xử lý
    """
    edges = detect_edges(processed_image)
    morph = apply_morphology(edges, kernel_size=(5, 5), iterations=2)
    plate_img, info, marked = detect_plate(processed_image, original_image)

    return {
        "edges_canny": edges,
        "morphology": morph,
        "detected": marked,
        "plate_cropped": plate_img,
        "plate_info": info,
    }
