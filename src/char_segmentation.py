"""
Module phân đoạn ký tự (Character Segmentation)
- Phân ngưỡng ảnh biển số
- Xử lý hình thái để tách ký tự
- Tìm contour từng ký tự
- Sắp xếp ký tự theo thứ tự đọc

Áp dụng kiến thức:
  Chương 5: Phân ngưỡng (Otsu, Adaptive Threshold), phân vùng
  Chương 7: Xử lý hình thái (Erosion, Opening)
"""

import cv2
import numpy as np


def binarize_plate(plate_image):
    """Nhị phân hóa ảnh biển số bằng phân ngưỡng.

    Áp dụng: Chương 5 - Phân ngưỡng
    - Otsu: Tự động tìm ngưỡng tối ưu dựa trên histogram
    - Adaptive: Tính ngưỡng cục bộ, tốt khi ánh sáng không đều

    Args:
        plate_image: Ảnh biển số (BGR hoặc grayscale)

    Returns:
        Ảnh nhị phân (trắng đen)
    """
    # Chuyển sang grayscale nếu cần
    if len(plate_image.shape) == 3:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_image.copy()

    # Lọc nhiễu nhẹ trước khi phân ngưỡng
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Phân ngưỡng Otsu (Chương 5)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return binary


def clean_binary_image(binary_image):
    """Làm sạch ảnh nhị phân bằng xử lý hình thái.

    Áp dụng: Chương 7 - Xử lý hình thái
    - Opening (Erosion → Dilation): Loại bỏ nhiễu nhỏ
    - Closing (Dilation → Erosion): Lấp lỗ nhỏ trong ký tự

    Args:
        binary_image: Ảnh nhị phân

    Returns:
        Ảnh nhị phân đã làm sạch
    """
    # Phần tử cấu trúc nhỏ để loại nhiễu
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Opening: loại bỏ nhiễu nhỏ (chấm, vết bẩn)
    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # Closing: lấp lỗ nhỏ bên trong ký tự
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    return cleaned


def find_char_contours(binary_image, plate_type="1_hang"):
    """Tìm contour của từng ký tự trên biển số.

    Áp dụng: Chương 5 - Phân vùng

    Args:
        binary_image: Ảnh nhị phân đã làm sạch
        plate_type: Loại biển số ("1_hang" hoặc "2_hang")

    Returns:
        Danh sách bounding box (x, y, w, h) của từng ký tự
    """
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_h, img_w = binary_image.shape[:2]
    char_bboxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        height_ratio = h / img_h
        width_ratio = w / img_w

        # Lọc contour: ký tự phải đủ lớn và có tỷ lệ hợp lý
        is_valid_height = 0.3 <= height_ratio <= 0.95   # Chiều cao ký tự so với biển số
        is_valid_width = 0.02 <= width_ratio <= 0.2      # Chiều rộng hợp lý
        is_valid_ratio = 0.1 <= aspect_ratio <= 1.0      # Tỷ lệ w/h của ký tự
        is_valid_area = w * h > 100                       # Diện tích tối thiểu

        if is_valid_height and is_valid_width and is_valid_ratio and is_valid_area:
            char_bboxes.append((x, y, w, h))

    return char_bboxes


def sort_characters(char_bboxes, plate_type="1_hang"):
    """Sắp xếp các ký tự theo thứ tự đọc.

    - Biển 1 hàng: sắp xếp từ trái sang phải (theo x)
    - Biển 2 hàng: chia thành 2 hàng (theo y), mỗi hàng sắp xếp theo x

    Args:
        char_bboxes: Danh sách bounding box ký tự
        plate_type: Loại biển số

    Returns:
        Danh sách bounding box đã sắp xếp theo thứ tự đọc
    """
    if not char_bboxes:
        return []

    if plate_type == "1_hang":
        # Biển 1 hàng: sắp xếp từ trái → phải
        return sorted(char_bboxes, key=lambda b: b[0])
    else:
        # Biển 2 hàng: chia theo hàng dựa vào tọa độ y
        if not char_bboxes:
            return []

        # Tính y trung bình của tất cả ký tự
        y_centers = [b[1] + b[3] // 2 for b in char_bboxes]
        y_mid = (min(y_centers) + max(y_centers)) / 2

        # Chia thành 2 hàng
        row1 = [b for b in char_bboxes if b[1] + b[3] // 2 < y_mid]
        row2 = [b for b in char_bboxes if b[1] + b[3] // 2 >= y_mid]

        # Mỗi hàng sắp xếp từ trái → phải
        row1.sort(key=lambda b: b[0])
        row2.sort(key=lambda b: b[0])

        return row1 + row2


def extract_char_images(plate_image, char_bboxes, target_size=(20, 30)):
    """Cắt và chuẩn hóa ảnh từng ký tự.

    Args:
        plate_image: Ảnh biển số (grayscale hoặc binary)
        char_bboxes: Danh sách bounding box ký tự
        target_size: Kích thước chuẩn hóa (width, height) cho mỗi ký tự

    Returns:
        Danh sách ảnh ký tự đã chuẩn hóa
    """
    # Đảm bảo ảnh là grayscale
    if len(plate_image.shape) == 3:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_image

    char_images = []
    for (x, y, w, h) in char_bboxes:
        # Cắt ký tự
        char_img = gray[y:y + h, x:x + w]

        # Resize về kích thước chuẩn
        char_resized = cv2.resize(char_img, target_size, interpolation=cv2.INTER_AREA)

        # Phân ngưỡng lại để đảm bảo ảnh binary rõ ràng
        _, char_binary = cv2.threshold(char_resized, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        char_images.append(char_binary)

    return char_images


def segment_characters(plate_image, plate_type="1_hang"):
    """Pipeline phân đoạn ký tự hoàn chỉnh.

    Quy trình:
        1. Nhị phân hóa biển số (Otsu - Chương 5)
        2. Làm sạch bằng morphology (Chương 7)
        3. Tìm contour ký tự (Chương 5)
        4. Sắp xếp thứ tự đọc
        5. Cắt và chuẩn hóa từng ký tự

    Args:
        plate_image: Ảnh biển số (BGR)
        plate_type: Loại biển số ("1_hang" hoặc "2_hang")

    Returns:
        tuple: (danh sách ảnh ký tự, danh sách bounding box, ảnh binary)
    """
    if plate_image is None or plate_image.size == 0:
        return [], [], None

    # Bước 1: Nhị phân hóa
    binary = binarize_plate(plate_image)

    # Bước 2: Làm sạch
    cleaned = clean_binary_image(binary)

    # Bước 3: Tìm contour ký tự
    char_bboxes = find_char_contours(cleaned, plate_type)

    # Bước 4: Sắp xếp
    sorted_bboxes = sort_characters(char_bboxes, plate_type)

    # Bước 5: Cắt ảnh ký tự
    char_images = extract_char_images(plate_image, sorted_bboxes)

    return char_images, sorted_bboxes, cleaned


def get_segmentation_steps(plate_image, plate_type="1_hang"):
    """Trả về ảnh ở từng bước phân đoạn (dùng để hiển thị minh họa).

    Returns:
        dict chứa ảnh ở mỗi bước xử lý
    """
    binary = binarize_plate(plate_image)
    cleaned = clean_binary_image(binary)
    char_bboxes = find_char_contours(cleaned, plate_type)
    sorted_bboxes = sort_characters(char_bboxes, plate_type)

    # Vẽ bounding box ký tự lên ảnh
    marked = plate_image.copy()
    for i, (x, y, w, h) in enumerate(sorted_bboxes):
        cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(marked, str(i), (x, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    return {
        "binary_otsu": binary,
        "morphology_cleaned": cleaned,
        "chars_marked": marked,
        "char_bboxes": sorted_bboxes,
    }
