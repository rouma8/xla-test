"""
Giao diện đơn giản - Tkinter
Chức năng: Chọn ảnh → Nhận dạng → Hiển thị kết quả
"""

import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from src.preprocess import preprocess_image
from src.plate_detector import detect_plate
from src.char_segmentation import segment_characters
from src.char_recognition import CharRecognizer, recognize_characters, format_plate_number


class LPRApp:
    """Giao diện nhận dạng biển số xe."""

    def __init__(self, root):
        self.root = root
        self.root.title("Nhan dang Bien so xe Viet Nam")
        self.root.geometry("900x650")
        self.root.resizable(True, True)

        # Khởi tạo KNN recognizer
        self.recognizer = CharRecognizer(k=5)
        self._load_model()

        self._build_ui()

    def _load_model(self):
        """Tải model KNN nếu có dataset."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        chars_dir = os.path.join(base_dir, "dataset", "chars")
        cls_path = os.path.join(base_dir, "dataset", "classifications.txt")
        flat_path = os.path.join(base_dir, "dataset", "flattened_images.txt")

        if os.path.isdir(chars_dir) and os.listdir(chars_dir):
            self.recognizer.train_from_folder(chars_dir)
        elif os.path.exists(cls_path) and os.path.exists(flat_path):
            self.recognizer.train_from_files(cls_path, flat_path)

    def _build_ui(self):
        """Xây dựng giao diện."""

        # === Tiêu đề ===
        title = tk.Label(self.root, text="NHAN DANG BIEN SO XE VIET NAM",
                         font=("Arial", 16, "bold"), pady=10)
        title.pack()

        # === Nút chức năng ===
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)

        btn_open = tk.Button(btn_frame, text="Chon anh", font=("Arial", 12),
                             command=self._open_image, padx=20, pady=5)
        btn_open.pack(side=tk.LEFT, padx=10)

        btn_run = tk.Button(btn_frame, text="Nhan dang", font=("Arial", 12),
                            command=self._run_recognition, padx=20, pady=5,
                            bg="#4CAF50", fg="white")
        btn_run.pack(side=tk.LEFT, padx=10)

        # === Khu vực hiển thị ảnh ===
        img_frame = tk.Frame(self.root)
        img_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # Ảnh gốc
        left_frame = tk.LabelFrame(img_frame, text="Anh goc", font=("Arial", 10))
        left_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        self.label_original = tk.Label(left_frame, text="Chua chon anh",
                                       bg="#f0f0f0", width=40, height=20)
        self.label_original.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        # Ảnh kết quả
        right_frame = tk.LabelFrame(img_frame, text="Ket qua", font=("Arial", 10))
        right_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        self.label_result_img = tk.Label(right_frame, text="Chua nhan dang",
                                         bg="#f0f0f0", width=40, height=20)
        self.label_result_img.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        # === Kết quả text ===
        result_frame = tk.Frame(self.root)
        result_frame.pack(pady=10)

        tk.Label(result_frame, text="Bien so:", font=("Arial", 12)).pack(side=tk.LEFT)
        self.label_result = tk.Label(result_frame, text="---",
                                     font=("Arial", 20, "bold"), fg="blue")
        self.label_result.pack(side=tk.LEFT, padx=10)

        # === Trạng thái ===
        self.label_status = tk.Label(self.root, text="San sang", font=("Arial", 9),
                                     fg="gray", anchor="w")
        self.label_status.pack(fill=tk.X, padx=10, pady=5)

        # Biến lưu trữ
        self.current_image_path = None
        self.current_image = None

    def _open_image(self):
        """Mở hộp thoại chọn ảnh."""
        path = filedialog.askopenfilename(
            title="Chon anh",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"),
                       ("All files", "*.*")]
        )
        if path:
            self.current_image_path = path
            self.current_image = cv2.imread(path)
            self._display_image(self.current_image, self.label_original)
            self.label_status.config(text=f"Da chon: {os.path.basename(path)}")
            self.label_result.config(text="---")

    def _display_image(self, cv_image, label_widget, max_size=(400, 400)):
        """Hiển thị ảnh OpenCV lên Tkinter Label."""
        if cv_image is None:
            return

        # Chuyển BGR → RGB
        if len(cv_image.shape) == 3:
            rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)

        # Resize giữ tỷ lệ
        h, w = rgb.shape[:2]
        scale = min(max_size[0] / w, max_size[1] / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(rgb, (new_w, new_h))

        # Chuyển sang ImageTk
        pil_img = Image.fromarray(resized)
        tk_img = ImageTk.PhotoImage(pil_img)

        label_widget.config(image=tk_img, text="")
        label_widget.image = tk_img  # Giữ reference

    def _run_recognition(self):
        """Chạy pipeline nhận dạng."""
        if self.current_image is None:
            messagebox.showwarning("Canh bao", "Vui long chon anh truoc!")
            return

        self.label_status.config(text="Dang xu ly...")
        self.root.update()

        try:
            # Bước 1: Tiền xử lý
            processed = preprocess_image(self.current_image)

            # Bước 2: Phát hiện biển số
            plate_img, plate_info, marked_img = detect_plate(processed, self.current_image)

            if plate_img is None:
                self.label_result.config(text="Khong tim thay", fg="red")
                self.label_status.config(text="Khong phat hien duoc bien so")
                self._display_image(self.current_image, self.label_result_img)
                return

            # Hiển thị ảnh có đánh dấu biển số
            self._display_image(marked_img, self.label_result_img)

            # Bước 3: Phân đoạn ký tự
            char_images, char_bboxes, binary = segment_characters(
                plate_img, plate_info["type"]
            )

            # Bước 4: Nhận dạng
            result = recognize_characters(char_images, self.recognizer)
            formatted = format_plate_number(result)

            self.label_result.config(text=formatted if formatted else result, fg="blue")
            self.label_status.config(
                text=f"Hoan tat! Loai: {plate_info['type']}, "
                     f"Tim thay {len(char_images)} ky tu"
            )

        except Exception as e:
            self.label_result.config(text="Loi!", fg="red")
            self.label_status.config(text=f"Loi: {str(e)}")
            messagebox.showerror("Loi", f"Co loi xay ra:\n{str(e)}")


def main():
    root = tk.Tk()
    app = LPRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
