import cv2
import numpy as np
import pytesseract
import os


# Uncomment the below line please if Tesseract is not in PATH (Windows)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# upload the image path in the bottom execution code to verify the image


# =====================================================
# ENTRY POINT
# =====================================================
def is_valid_text_image(image_path):
    return load_image(image_path)

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None: return -1, "Image not readable"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    data = {"image": image, "gray": gray, "blurred": blurred}
    return check_blur(data)

def check_blur(data):
    gray = data["gray"]
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_energy = np.mean(gx * gx + gy * gy)
    if lap_var < 11 and grad_energy < 30: return -1, "Rejected: image is blurry"
    return check_contrast(data)

def check_contrast(data):
    if data["blurred"].std() < 8: return -1, "Rejected: low contrast"
    return check_brightness(data)

def check_brightness(data):
    gray = data["gray"]
    bright_ratio = np.mean(gray > 245)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size
    if bright_ratio > 0.60 and edge_ratio < 0.001: return -1, "Rejected: overexposed"
    return check_blank(data)

def check_blank(data):
    if (np.count_nonzero(data["gray"]) / data["gray"].size) < 0.008: return -1, "Rejected: blank"
    return check_document_likeness(data)

def check_document_likeness(data):
    hsv = cv2.cvtColor(data["image"], cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:, :, 1])
    color_variation = np.mean([np.std(data["image"][:,:,i]) for i in range(3)])
    edges = cv2.Canny(data["gray"], 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size
    # Strict photo rejection
    if saturation > 85 and color_variation > 45 and edge_ratio < 0.008:
        return -1, "Rejected: natural photo"
    return check_human_animal_presence(data)

def check_human_animal_presence(data):
    gray = data["gray"]
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if len(face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))) > 0:
        return -1, "Rejected: human presence"
    animal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface.xml")
    if len(animal_cascade.detectMultiScale(gray, 1.1, 4, minSize=(80, 80))) > 0:
        return -1, "Rejected: animal presence"
    return check_text_dominance(data)

# =====================================================
# (TO PASS HANDWRITING / REJECT MEMES)
# =====================================================
def check_text_dominance(data):
    gray, (h, w) = data["gray"], data["gray"].shape
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    large_objects = 0
    text_like_elements = 0
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        ratio = (cw * ch) / (h * w)
        if ratio > 0.25: large_objects += 1  # Meme background/Person
        elif 10 < (cw * ch) < (h * w * 0.05): text_like_elements += 1 # Likely letters

    # Rejects memes/scene photos with one big central object and no surrounding text
    if large_objects >= 1 and text_like_elements < 20:
        return -1, "Rejected: non-text dominant scene or poster"
    
    data["text_elements"] = text_like_elements
    return check_edges(data)

def check_edges(data):
    edges = cv2.Canny(data["blurred"], 30, 120)
    edge_ratio = np.sum(edges > 0) / edges.size
    if edge_ratio < 0.0015: return -1, "Rejected: no text structure"
    
    _, binary = cv2.threshold(data["blurred"], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    data["morph"] = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return check_ink(data)

def check_ink(data):
    ink_ratio = np.count_nonzero(data["morph"] == 0) / data["morph"].size
    if ink_ratio < 0.012: return -1, "Rejected: low ink density"
    data["ink_ratio"] = ink_ratio
    return check_ocr(data)

def check_ocr(data):
    # Use PSM 11 (Sparse text) which is better for varied handwriting
    ocr_data = pytesseract.image_to_data(data["morph"], output_type=pytesseract.Output.DICT, config="--psm 11")
    words = [w for i, w in enumerate(ocr_data["text"]) if int(ocr_data["conf"][i]) > 20 and w.strip()]
    data["word_count"] = len(words)
    data["char_count"] = sum(len(w) for w in words)
    return final_decision(data)

def final_decision(data):
    # Passes if we have a lot of letters (Print) OR complex stroke patterns (Handwriting)
    if (data["word_count"] >= 5 or data["char_count"] >= 35) or \
       (data["text_elements"] > 50 and data["ink_ratio"] > 0.02):
        return 1, "✅ VALID TEXT IMAGE"
    return -1, "Rejected: insufficient text content"




# ================================
# MAIN INPUT + EXECUTION CODE
# ================================
if __name__ == "__main__":
    folder_path = ''
    folder_items = os.listdir(folder_path)
    if folder_items is not None:
        for item in folder_items:
            if item.lower().endswith(('jpg', 'png', 'heic', 'jpeg')):
                image_path = os.path.join(folder_path,item)
                print(f"\n--- Analyzing image: {image_path} ---")
                result = is_valid_text_image(image_path)

                if result == 1:
                    print(f'{item}  is VALID')
                else:
                    print(f'{item} is INVALID')
            else:
                print(f'{item} Not valid image extention')        