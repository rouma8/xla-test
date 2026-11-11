import cv2, os, sys
sys.path.insert(0, ".")
from src.preprocess import preprocess_image
from src.plate_detector import detect_plate
from src.char_segmentation import segment_characters
from src.char_recognition import CharRecognizer, recognize_characters, format_plate_number

img_path = "dataset/plates/1.jpg"
image = cv2.imread(img_path)
print("Anh: {}, Kich thuoc: {}".format(img_path, image.shape))

# Buoc 1
processed = preprocess_image(image)
print("[1] Tien xu ly: OK ({})".format(processed.shape))

# Buoc 2
plate_img, plate_info, marked_img = detect_plate(processed, image)
if plate_img is not None:
    print("[2] Phat hien bien so: OK - Loai: {}, Bbox: {}".format(plate_info["type"], plate_info["bbox"]))
else:
    print("[2] Khong tim thay bien so!")
    sys.exit()

# Buoc 3
char_images, char_bboxes, binary = segment_characters(plate_img, plate_info["type"])
print("[3] Phan doan ky tu: Tim thay {} ky tu".format(len(char_images)))

# Buoc 4
recognizer = CharRecognizer(k=5)
cls_path = os.path.join("dataset", "classifications.txt")
flat_path = os.path.join("dataset", "flattened_images.txt")
if os.path.exists(cls_path):
    recognizer.train_from_files(cls_path, flat_path)

result = recognize_characters(char_images, recognizer)
formatted = format_plate_number(result)
print("[4] Nhan dang: {}".format(result))
print("    Dinh dang: {}".format(formatted))
print()
print("=== PIPELINE CHAY THANH CONG! ===")
