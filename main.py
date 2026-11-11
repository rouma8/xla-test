"""
Main Pipeline - Nhận dạng biển số xe Việt Nam

Chạy pipeline hoàn chỉnh:
    Ảnh đầu vào → Tiền xử lý → Phát hiện biển số
    → Phân đoạn ký tự → Nhận dạng → Kết quả

Sử dụng: python main.py -i <đường_dẫn_ảnh>
"""

import cv2
import argparse
import os
import matplotlib.pyplot as plt

from src.preprocess import preprocess_image, get_preprocessing_steps
from src.plate_detector import detect_plate, get_detection_steps
from src.char_segmentation import segment_characters, get_segmentation_steps
from src.char_recognition import CharRecognizer, recognize_characters, format_plate_number


def run_pipeline(image_path, chars_dataset=None, show_steps=True):
    """Chạy toàn bộ pipeline nhận dạng biển số.

    Args:
        image_path: Đường dẫn ảnh đầu vào
        chars_dataset: Đường dẫn thư mục dataset ký tự (để train KNN)
        show_steps: Hiển thị ảnh từng bước xử lý

    Returns:
        str: Biển số nhận dạng được
    """
    # === Đọc ảnh ===
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Loi] Khong doc duoc anh: {image_path}")
        return None

    print(f"\n{'='*50}")
    print(f"  NHAN DANG BIEN SO XE")
    print(f"  Anh: {os.path.basename(image_path)}")
    print(f"  Kich thuoc: {image.shape[1]}x{image.shape[0]}")
    print(f"{'='*50}\n")

    # === Bước 1: Tiền xử lý ===
    print("[1/4] Tien xu ly anh...")
    processed = preprocess_image(image)

    # === Bước 2: Phát hiện biển số ===
    print("[2/4] Phat hien bien so...")
    plate_img, plate_info, marked_img = detect_plate(processed, image)

    if plate_img is None:
        print("[!] Khong tim thay bien so trong anh!")
        if show_steps:
            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title("Khong tim thay bien so")
            plt.axis("off")
            plt.show()
        return None

    print(f"    -> Tim thay bien so loai: {plate_info['type']}")
    print(f"    -> Vi tri: x={plate_info['bbox'][0]}, y={plate_info['bbox'][1]}")
    print(f"    -> Kich thuoc: {plate_info['bbox'][2]}x{plate_info['bbox'][3]}")

    # === Bước 3: Phân đoạn ký tự ===
    print("[3/4] Phan doan ky tu...")
    char_images, char_bboxes, binary_plate = segment_characters(
        plate_img, plate_info["type"]
    )
    print(f"    -> Tim thay {len(char_images)} ky tu")

    # === Bước 4: Nhận dạng ký tự ===
    print("[4/4] Nhan dang ky tu...")

    # Khởi tạo bộ nhận dạng KNN
    recognizer = CharRecognizer(k=5)

    # Train từ dataset nếu có
    if chars_dataset and os.path.isdir(chars_dataset):
        recognizer.train_from_folder(chars_dataset)
    else:
        # Thử load từ file classifications
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cls_path = os.path.join(base_dir, "dataset", "classifications.txt")
        flat_path = os.path.join(base_dir, "dataset", "flattened_images.txt")
        if os.path.exists(cls_path) and os.path.exists(flat_path):
            recognizer.train_from_files(cls_path, flat_path)
        else:
            print("    [!] Chua co dataset huan luyen KNN")
            print("    [!] Ket qua se la '?' cho moi ky tu")

    result = recognize_characters(char_images, recognizer)
    formatted = format_plate_number(result)

    print(f"\n{'='*50}")
    print(f"  KET QUA: {result}")
    print(f"  DINH DANG: {formatted}")
    print(f"{'='*50}\n")

    # === Hiển thị kết quả ===
    if show_steps:
        show_pipeline_steps(image, processed, marked_img, plate_img,
                           binary_plate, char_images, char_bboxes, result)

    return formatted


def show_pipeline_steps(original, processed, marked, plate, binary_plate,
                        char_images, char_bboxes, result):
    """Hiển thị ảnh ở từng bước xử lý bằng matplotlib."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("PIPELINE NHAN DANG BIEN SO XE", fontsize=16, fontweight="bold")

    # 1. Ảnh gốc
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Anh goc")
    axes[0, 0].axis("off")

    # 2. Ảnh tiền xử lý
    axes[0, 1].imshow(processed, cmap="gray")
    axes[0, 1].set_title("2. Tien xu ly\n(Grayscale + Loc nhieu + Histogram)")
    axes[0, 1].axis("off")

    # 3. Phát hiện biển số
    axes[0, 2].imshow(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("3. Phat hien bien so\n(Canny + Morphology + Contour)")
    axes[0, 2].axis("off")

    # 4. Biển số cắt ra
    if plate is not None:
        axes[1, 0].imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("4. Bien so cat ra")
    axes[1, 0].axis("off")

    # 5. Ảnh nhị phân + ký tự
    if binary_plate is not None:
        axes[1, 1].imshow(binary_plate, cmap="gray")
    axes[1, 1].set_title("5. Phan nguong + Ky tu\n(Otsu + Morphology)")
    axes[1, 1].axis("off")

    # 6. Kết quả
    if char_images:
        # Ghép ảnh các ký tự thành 1 dải
        display_chars = []
        for img in char_images:
            resized = cv2.resize(img, (30, 45))
            # Thêm viền trắng giữa các ký tự
            bordered = cv2.copyMakeBorder(resized, 2, 2, 2, 2,
                                          cv2.BORDER_CONSTANT, value=128)
            display_chars.append(bordered)
        if display_chars:
            char_strip = np.hstack(display_chars)
            axes[1, 2].imshow(char_strip, cmap="gray")
    axes[1, 2].set_title(f"6. Ket qua: {result}")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nhan dang bien so xe Viet Nam")
    parser.add_argument("-i", "--image", required=True, help="Duong dan anh dau vao")
    parser.add_argument("-d", "--dataset", default="dataset/chars",
                        help="Duong dan thu muc dataset ky tu")
    parser.add_argument("--no-show", action="store_true",
                        help="Khong hien thi anh ket qua")

    args = parser.parse_args()

    run_pipeline(
        image_path=args.image,
        chars_dataset=args.dataset,
        show_steps=not args.no_show,
    )
