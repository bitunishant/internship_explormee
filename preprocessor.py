import cv2
import numpy as np
import pytesseract
import os


# Uncomment the below line please if Tesseract is not in PATH (Windows)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# upload the image path in the bottom execution code to verify the image



def is_valid_text_image(image_path):
    """
    Returns:
        1  -> readable written / printed text present
       -1  -> invalid image (photo / blank / decorative / tiny text)
    """

    # =====================================================
    # LOAD IMAGE
    # =====================================================
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Fail (Load)")
        return -1

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # =====================================================
    # IMAGE QUALITY CHECKS
    # =====================================================

    # --- Blur check (focus)
    blur_score = cv2.Laplacian(blurred, cv2.CV_64F).var()
    if blur_score < 18:
        print(f"❌ Fail (Blur): {blur_score:.2f}")
        return -1

    # --- Brightness check
    brightness = blurred.mean()
    if brightness < 15 or brightness > 245:
        print(f"❌ Fail (Brightness): {brightness:.2f}")
        return -1

    # --- Contrast check
    contrast = blurred.std()
    if contrast < 9:
        print(f"❌ Fail (Contrast): {contrast:.2f}")
        return -1

    # =====================================================
    # BLANK IMAGE CHECK
    # =====================================================
    non_zero_ratio = np.count_nonzero(gray) / gray.size
    if non_zero_ratio < 0.01:
        print(f"❌ Fail (Blank): {non_zero_ratio:.4f}")
        return -1

    # =====================================================
    # TEXT STRUCTURE CHECK
    # =====================================================

    # --- Binary thresholding
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --- Morphological closing (connect characters)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # --- Edge density (text creates edges)
    edges = cv2.Canny(blurred, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size
    if edge_ratio < 0.002:
        print(f"❌ Fail (Edges): {edge_ratio:.4f}")
        return -1

    # =====================================================
    # INK COVERAGE CHECK
    # =====================================================
    ink_pixels = np.count_nonzero(morph == 0)
    ink_ratio = ink_pixels / morph.size

    if ink_ratio < 0.025:
        print(f"❌ Fail (Ink Coverage): {ink_ratio:.4f}")
        return -1

    # =====================================================
    # RULED / DECORATIVE PAGE CHECK
    # =====================================================
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    horizontal_strength = np.mean(np.abs(sobel_y))
    vertical_strength = np.mean(np.abs(sobel_x))

    # Reject ruled pages only when ink is low
    if horizontal_strength > 2.5 * vertical_strength and ink_ratio < 0.04:
        print(
            f"❌ Fail (Decorative Ruled Page): "
            f"h={horizontal_strength:.2f}, "
            f"v={vertical_strength:.2f}, "
            f"ink_ratio={ink_ratio:.3f}"
        )
        return -1

    # =====================================================
    # OCR STAGE 1 — CONFIDENT WORDS
    # =====================================================
    ocr_data = pytesseract.image_to_data(
        morph,
        output_type=pytesseract.Output.DICT,
        config="--psm 6"
    )

    word_count = 0
    confident_characters = 0
    confidence_sum = 0

    for i, conf_str in enumerate(ocr_data["conf"]):
        if conf_str == "-1":
            continue

        confidence = int(conf_str)
        text = ocr_data["text"][i].strip()

        if confidence >= 30 and text:
            word_count += 1
            confident_characters += len(text)
            confidence_sum += confidence

    # =====================================================
    # OCR STAGE 2 — RAW TEXT FALLBACK
    # =====================================================
    raw_text = pytesseract.image_to_string(morph, config="--psm 6")
    raw_characters = len(raw_text.replace(" ", "").replace("\n", ""))

    # =====================================================
    # FINAL OCR DECISION
    # =====================================================
    if not (
        word_count >= 3
        or confident_characters >= 8
        or raw_characters >= 25
    ):
        print(
            f"❌ Fail (OCR Count): "
            f"words={word_count}, "
            f"conf_chars={confident_characters}, "
            f"raw_chars={raw_characters}"
        )
        return -1

    average_confidence = confidence_sum / max(word_count, 1)
    if average_confidence < 35:
        print(f"❌ Fail (OCR Confidence): {average_confidence:.2f}")
        return -1

    # =====================================================
    # ✅ SUCCESS
    # =====================================================
    print(
        f"✅ SUCCESS | words={word_count}, "
        f"conf_chars={confident_characters}, "
        f"raw_chars={raw_characters}, "
        f"ink_ratio={ink_ratio:.3f}, "
        f"avg_conf={average_confidence:.2f}"
    )
    return 1



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