"""
Module tiền xử lý ảnh (Preprocessing)
- Chuyển đổi ảnh sang grayscale
- Lọc nhiễu (Gaussian, Bilateral, Median)
- Cân bằng histogram (Histogram Equalization)
- Chuyển đổi hệ màu

Áp dụng kiến thức:
  Chương 3: Nâng cao chất lượng ảnh (lọc, histogram)
  Chương 4: Các hệ màu (RGB, Grayscale, HSV)
"""

import cv2
import numpy as np


def convert_to_grayscale(image):
    """Chuyển ảnh BGR sang grayscale.

    Áp dụng: Chương 4 - Chuyển đổi hệ màu
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def apply_bilateral_filter(image, d=11, sigma_color=17, sigma_space=17):
    """Lọc nhiễu bằng Bilateral Filter.

    Ưu điểm: Giảm nhiễu nhưng giữ được biên (edge-preserving).
    Áp dụng: Chương 3 - Lọc nhiễu

    Args:
        image: Ảnh grayscale đầu vào
        d: Đường kính vùng lân cận
        sigma_color: Sigma trong không gian màu
        sigma_space: Sigma trong không gian tọa độ
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """Lọc nhiễu bằng Gaussian Blur.

    Áp dụng: Chương 3 - Biến đổi cửa sổ di chuyển (convolution)

    Args:
        image: Ảnh đầu vào
        kernel_size: Kích thước kernel
    """
    return cv2.GaussianBlur(image, kernel_size, 0)


def apply_median_filter(image, kernel_size=5):
    """Lọc nhiễu bằng bộ lọc trung vị.

    Hiệu quả với nhiễu muối tiêu (salt-and-pepper noise).
    Áp dụng: Chương 3 - Lọc trung vị

    Args:
        image: Ảnh đầu vào
        kernel_size: Kích thước kernel (số lẻ)
    """
    return cv2.medianBlur(image, kernel_size)


def equalize_histogram(image):
    """Cân bằng histogram để tăng độ tương phản.

    Áp dụng: Chương 3 - Cân bằng tần suất (Histogram Equalization)
    """
    return cv2.equalizeHist(image)


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Cân bằng histogram thích ứng (CLAHE).

    Cải tiến của histogram equalization, xử lý cục bộ thay vì toàn cục.
    Áp dụng: Chương 3 - Cân bằng tần suất nâng cao

    Args:
        image: Ảnh grayscale
        clip_limit: Ngưỡng giới hạn tương phản
        tile_grid_size: Kích thước lưới chia ảnh
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def preprocess_image(image):
    """Pipeline tiền xử lý hoàn chỉnh.

    Quy trình:
        1. Chuyển sang grayscale (Chương 4)
        2. Lọc nhiễu bilateral (Chương 3)
        3. Cân bằng histogram CLAHE (Chương 3)

    Args:
        image: Ảnh BGR đầu vào

    Returns:
        Ảnh grayscale đã được tiền xử lý
    """
    # Bước 1: Chuyển sang ảnh xám
    gray = convert_to_grayscale(image)

    # Bước 2: Lọc nhiễu giữ biên
    filtered = apply_bilateral_filter(gray)

    # Bước 3: Cân bằng histogram thích ứng
    enhanced = apply_clahe(filtered)

    return enhanced


def get_preprocessing_steps(image):
    """Trả về ảnh ở từng bước tiền xử lý (dùng để hiển thị minh họa).

    Returns:
        dict chứa ảnh ở mỗi bước xử lý
    """
    gray = convert_to_grayscale(image)
    filtered = apply_bilateral_filter(gray)
    enhanced = apply_clahe(filtered)

    return {
        "original": image,
        "grayscale": gray,
        "filtered": filtered,
        "enhanced": enhanced,
    }
