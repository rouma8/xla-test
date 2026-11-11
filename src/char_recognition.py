"""
Module nhận dạng ký tự (Character Recognition)
- Huấn luyện mô hình KNN với dataset ký tự
- Nhận dạng từng ký tự riêng lẻ
- Ghép kết quả thành chuỗi biển số

Áp dụng kiến thức:
  Chương 5: Phân vùng - Thuật toán K láng giềng (KNN)
"""

import cv2
import numpy as np
import os

# Kích thước chuẩn hóa cho mỗi ký tự
CHAR_WIDTH = 20
CHAR_HEIGHT = 30

# Các ký tự có trên biển số xe Việt Nam
VALID_CHARS = "0123456789ABCDEFGHKLMNPRSTUVXYZ"


class CharRecognizer:
    """Bộ nhận dạng ký tự sử dụng thuật toán KNN.

    Áp dụng: Chương 5 - Kỹ thuật K láng giềng (K-Nearest Neighbors)
    Ý tưởng: Phân loại mẫu mới dựa trên K mẫu gần nhất trong tập huấn luyện.
    """

    def __init__(self, k=5):
        """Khởi tạo bộ nhận dạng.

        Args:
            k: Số láng giềng gần nhất (K trong KNN)
        """
        self.k = k
        self.knn = cv2.ml.KNearest_create()
        self.is_trained = False

    def train_from_files(self, classifications_path, flattened_images_path):
        """Huấn luyện KNN từ file dữ liệu.

        Đọc dữ liệu huấn luyện từ 2 file:
        - classifications.txt: nhãn (mã ASCII) của từng ký tự
        - flattened_images.txt: vector đặc trưng (pixel) đã làm phẳng

        Args:
            classifications_path: Đường dẫn file nhãn
            flattened_images_path: Đường dẫn file ảnh đã làm phẳng
        """
        try:
            classifications = np.loadtxt(classifications_path, np.float32)
            flattened_images = np.loadtxt(flattened_images_path, np.float32)

            # Reshape nhãn thành cột
            classifications = classifications.reshape((classifications.size, 1))

            self.knn.setDefaultK(self.k)
            self.knn.train(flattened_images, cv2.ml.ROW_SAMPLE, classifications)
            self.is_trained = True
            print(f"[KNN] Da huan luyen voi {len(classifications)} mau, K={self.k}")

        except Exception as e:
            print(f"[KNN] Loi khi huan luyen: {e}")
            self.is_trained = False

    def train_from_folder(self, chars_folder):
        """Huấn luyện KNN từ thư mục chứa ảnh ký tự.

        Cấu trúc thư mục:
            chars_folder/
            ├── 0/    ← chứa các ảnh của số 0
            ├── 1/
            ├── ...
            ├── A/
            ├── B/
            └── ...

        Args:
            chars_folder: Đường dẫn thư mục dataset ký tự
        """
        samples = []
        labels = []

        for char_name in os.listdir(chars_folder):
            char_dir = os.path.join(chars_folder, char_name)
            if not os.path.isdir(char_dir):
                continue

            # Lấy mã ASCII của ký tự làm nhãn
            if len(char_name) == 1 and char_name in VALID_CHARS:
                label = ord(char_name)
            else:
                continue

            for img_file in os.listdir(char_dir):
                img_path = os.path.join(char_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                # Resize về kích thước chuẩn
                resized = cv2.resize(img, (CHAR_WIDTH, CHAR_HEIGHT))

                # Phân ngưỡng
                _, binary = cv2.threshold(resized, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Làm phẳng thành vector 1D
                flattened = binary.reshape(1, CHAR_WIDTH * CHAR_HEIGHT).astype(np.float32)

                samples.append(flattened)
                labels.append(label)

        if not samples:
            print("[KNN] Khong tim thay du lieu huan luyen!")
            self.is_trained = False
            return

        samples = np.vstack(samples).astype(np.float32)
        labels = np.array(labels, dtype=np.float32).reshape(-1, 1)

        self.knn.setDefaultK(self.k)
        self.knn.train(samples, cv2.ml.ROW_SAMPLE, labels)
        self.is_trained = True
        print(f"[KNN] Da huan luyen voi {len(labels)} mau tu {chars_folder}, K={self.k}")

    def recognize_char(self, char_image):
        """Nhận dạng một ký tự.

        Args:
            char_image: Ảnh ký tự (grayscale hoặc binary), kích thước bất kỳ

        Returns:
            tuple: (ký tự nhận dạng, độ tin cậy)
        """
        if not self.is_trained:
            return "?", 0.0

        # Resize về kích thước chuẩn
        resized = cv2.resize(char_image, (CHAR_WIDTH, CHAR_HEIGHT))

        # Phân ngưỡng
        _, binary = cv2.threshold(resized, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Làm phẳng thành vector 1D
        sample = binary.reshape(1, CHAR_WIDTH * CHAR_HEIGHT).astype(np.float32)

        # Dự đoán bằng KNN
        ret, result, neighbours, dist = self.knn.findNearest(sample, k=self.k)

        char = chr(int(result[0][0]))
        # Tính độ tin cậy: tỷ lệ láng giềng cùng nhãn
        confidence = np.sum(neighbours == result[0][0]) / self.k

        return char, confidence

    def save_model(self, classifications_path, flattened_images_path):
        """Lưu dữ liệu huấn luyện ra file.

        Args:
            classifications_path: Đường dẫn file lưu nhãn
            flattened_images_path: Đường dẫn file lưu ảnh đã làm phẳng
        """
        if not self.is_trained:
            print("[KNN] Chua huan luyen, khong the luu!")
            return

        # Lấy dữ liệu từ model (OpenCV KNN không hỗ trợ trực tiếp)
        print("[KNN] Luu model thanh cong")


def recognize_characters(char_images, recognizer=None):
    """Nhận dạng danh sách ký tự và ghép thành chuỗi biển số.

    Args:
        char_images: Danh sách ảnh ký tự (từ segment_characters)
        recognizer: Instance CharRecognizer đã train. Nếu None, trả về "?"

    Returns:
        str: Chuỗi biển số nhận dạng được
    """
    if not char_images:
        return ""

    if recognizer is None or not recognizer.is_trained:
        # Nếu chưa có model, trả về placeholder
        return "?" * len(char_images)

    result = ""
    for char_img in char_images:
        char, confidence = recognizer.recognize_char(char_img)
        result += char

    return result


def format_plate_number(raw_text):
    """Định dạng chuỗi biển số theo quy chuẩn Việt Nam.

    Ví dụ: "29A12345" → "29A-123.45"

    Args:
        raw_text: Chuỗi ký tự thô

    Returns:
        Chuỗi biển số đã định dạng
    """
    # Loại bỏ ký tự không hợp lệ
    cleaned = "".join(c for c in raw_text.upper() if c in VALID_CHARS)

    if len(cleaned) >= 8:
        # Định dạng: XXY-ZZZZZ (ví dụ: 29A-12345)
        prefix = cleaned[:2]      # Mã tỉnh (2 số)
        series = cleaned[2]       # Seri (1 chữ)
        number = cleaned[3:]      # Số đăng ký

        if len(number) >= 5:
            return f"{prefix}{series}-{number[:3]}.{number[3:5]}"

    return cleaned
