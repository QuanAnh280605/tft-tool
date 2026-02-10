"""
OCR Service Module

Provides text recognition from images using EasyOCR and Tesseract.
Optimized for reading champion names from TFT game screenshots.
Uses singleton pattern to load the reader only once at startup.
"""

import logging
import os

import cv2
import easyocr
import numpy as np
import pytesseract
from PIL import Image

# Auto-detect Tesseract path on Windows
_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\admin\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
]
for _path in _TESSERACT_PATHS:
    if os.path.exists(_path):
        pytesseract.pytesseract.tesseract_cmd = _path
        break

logger = logging.getLogger(__name__)

# ─── Singleton Model Loader ────────────────────────────────────
_reader = None

SUPPORTED_LANGUAGES = ["en"]


def _load_reader():
    """Load EasyOCR reader (only once)."""
    global _reader

    if _reader is None:
        logger.info("Loading EasyOCR reader (languages: %s) ...", SUPPORTED_LANGUAGES)
        _reader = easyocr.Reader(SUPPORTED_LANGUAGES, gpu=True)
        logger.info("EasyOCR reader loaded successfully.")

    return _reader


def get_reader():
    """Get the loaded EasyOCR reader."""
    return _load_reader()


# ─── Core OCR Functions ────────────────────────────────────────

def recognize_text(image) -> str:
    """
    Recognize text from a single image.

    Args:
        image: PIL.Image, numpy array (BGR or RGB), or file path string.

    Returns:
        Recognized text string (all detected text joined).
    """
    reader = get_reader()
    img_array = _to_numpy(image)

    results = reader.readtext(img_array, detail=0)
    text = " ".join(results).strip()

    return text


def recognize_text_detail(image) -> list[dict]:
    """
    Recognize text with bounding boxes and confidence scores.

    Args:
        image: PIL.Image, numpy array (BGR or RGB), or file path string.

    Returns:
        List of dicts with 'text', 'confidence', and 'bbox' keys.
    """
    reader = get_reader()
    img_array = _to_numpy(image)

    results = reader.readtext(img_array)

    detections = []
    for bbox, text, confidence in results:
        detections.append({
            "text": text,
            "confidence": round(float(confidence), 4),
            "bbox": {
                "top_left": [int(bbox[0][0]), int(bbox[0][1])],
                "bottom_right": [int(bbox[2][0]), int(bbox[2][1])],
            },
        })

    return detections


def recognize_name_strip(
    shop_region: np.ndarray,
    num_slots: int = 5,
    name_ratio: float = 0.2,
) -> list[dict]:
    """
    Optimized OCR for TFT shop: crop bottom name area of each slot,
    concatenate into a single horizontal strip, and OCR once.

    This is ~3-5x faster than calling OCR per slot individually.

    Args:
        shop_region: numpy array (BGR) of the shop region.
        num_slots: number of champion slots (default 5).
        name_ratio: fraction of slot height for the name area (default 0.2 = bottom 1/5).

    Returns:
        List of dicts with 'slot', 'text', 'confidence', and 'bbox' keys.
    """
    reader = get_reader()
    shop_h, shop_w = shop_region.shape[:2]
    slot_w = shop_w // num_slots

    # 1. Crop bottom name region from each slot
    name_y_start = int(shop_h * (1 - name_ratio))
    name_regions = []
    slot_boundaries = []  # Track x-offset of each slot in the strip

    separator_w = 10  # White separator between slots for better OCR separation
    current_x = 0

    for i in range(num_slots):
        sx1 = i * slot_w
        sx2 = (i + 1) * slot_w if i < num_slots - 1 else shop_w

        name_img = shop_region[name_y_start:, sx1:sx2]
        name_regions.append(name_img)
        slot_boundaries.append((current_x, current_x + name_img.shape[1]))
        current_x += name_img.shape[1] + separator_w

    # 2. Build single horizontal strip with white separators
    strip_h = max(r.shape[0] for r in name_regions)
    strip_w = sum(r.shape[1] for r in name_regions) + separator_w * (num_slots - 1)

    strip = np.ones((strip_h, strip_w, 3), dtype=np.uint8) * 255  # White background
    x_offset = 0
    for i, region in enumerate(name_regions):
        rh, rw = region.shape[:2]
        strip[:rh, x_offset:x_offset + rw] = region
        x_offset += rw + separator_w

    # 3. Single OCR call on the entire strip
    strip_rgb = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
    raw_results = reader.readtext(strip_rgb)

    # 4. Map OCR results back to slots by x-coordinate
    results = [{"slot": i + 1, "text": "", "confidence": 0.0} for i in range(num_slots)]

    for bbox, text, confidence in raw_results:
        # Use center-x of the detected text to determine which slot
        text_center_x = (bbox[0][0] + bbox[2][0]) / 2

        for i, (bx_start, bx_end) in enumerate(slot_boundaries):
            if bx_start <= text_center_x < bx_end:
                # Append text if slot already has text (multi-word names)
                if results[i]["text"]:
                    results[i]["text"] += " " + text
                else:
                    results[i]["text"] = text
                results[i]["confidence"] = max(
                    results[i]["confidence"],
                    round(float(confidence), 4),
                )
                break

    # 5. Fallback: re-OCR individual slots that got empty text
    #    (handles short names like "Vi" that the strip OCR may miss)
    for i in range(num_slots):
        if not results[i]["text"]:
            slot_img = name_regions[i]
            slot_rgb = cv2.cvtColor(slot_img, cv2.COLOR_BGR2RGB)

            # Scale up small images for better detection
            sh, sw = slot_rgb.shape[:2]
            if sh < 40:
                scale = 40 / sh
                slot_rgb = cv2.resize(
                    slot_rgb, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )

            slot_results = reader.readtext(
                slot_rgb, text_threshold=0.3, low_text=0.2,
            )
            if slot_results:
                texts = [t for _, t, _ in slot_results]
                confs = [c for _, _, c in slot_results]
                results[i]["text"] = " ".join(texts)
                results[i]["confidence"] = round(float(max(confs)), 4)
                logger.info(
                    "🔄 Fallback OCR for slot %d: '%s' (conf=%.2f)",
                    i + 1, results[i]["text"], results[i]["confidence"],
                )

    # 6. Add bbox info (relative to shop_region)
    for i in range(num_slots):
        sx1 = i * slot_w
        sx2 = (i + 1) * slot_w if i < num_slots - 1 else shop_w
        results[i]["bbox"] = {
            "x1": sx1, "y1": name_y_start, "x2": sx2, "y2": shop_h,
        }

    return results


def recognize_name_strip_tesseract(
    shop_region: np.ndarray,
    num_slots: int = 5,
    name_ratio: float = 0.2,
) -> list[dict]:
    """
    Fast OCR for TFT shop using Tesseract.
    Crops the bottom name area of each slot and OCRs with Tesseract.

    Tesseract is ~5-10x faster than EasyOCR for simple printed text.

    Args:
        shop_region: numpy array (BGR) of the shop region.
        num_slots: number of champion slots (default 5).
        name_ratio: fraction of slot height for the name area (default 0.2).

    Returns:
        List of dicts with 'slot', 'text', 'confidence', and 'bbox' keys.
    """
    shop_h, shop_w = shop_region.shape[:2]
    slot_w = shop_w // num_slots
    name_y_start = int(shop_h * (1 - name_ratio))

    # Tesseract config: single line mode, whitelist letters only
    tess_config = (
        "--oem 3 --psm 7 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
    )

    results = []
    for i in range(num_slots):
        sx1 = i * slot_w
        sx2 = (i + 1) * slot_w if i < num_slots - 1 else shop_w

        name_img = shop_region[name_y_start:, sx1:sx2]

        # Preprocess for Tesseract: grayscale + threshold for better accuracy
        gray = cv2.cvtColor(name_img, cv2.COLOR_BGR2GRAY)
        # Invert if dark background (game UI) — Tesseract expects dark text on light bg
        if gray.mean() < 128:
            gray = cv2.bitwise_not(gray)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Scale up small images for better recognition
        h_name, w_name = binary.shape[:2]
        if h_name < 32:
            scale = 32 / h_name
            binary = cv2.resize(
                binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
            )

        pil_img = Image.fromarray(binary)

        try:
            # Get text + confidence
            data = pytesseract.image_to_data(
                pil_img, config=tess_config, output_type=pytesseract.Output.DICT,
            )
            # Extract words with confidence > 0
            words = []
            confidences = []
            for j, conf in enumerate(data["conf"]):
                conf_val = int(conf)
                if conf_val > 0 and data["text"][j].strip():
                    words.append(data["text"][j].strip())
                    confidences.append(conf_val)

            text = " ".join(words)
            avg_conf = sum(confidences) / len(confidences) / 100 if confidences else 0.0
        except Exception as e:
            logger.warning("Tesseract OCR failed for slot %d: %s", i + 1, e)
            text = ""
            avg_conf = 0.0

        results.append({
            "slot": i + 1,
            "text": text,
            "confidence": round(avg_conf, 4),
            "bbox": {"x1": sx1, "y1": name_y_start, "x2": sx2, "y2": shop_h},
        })

    return results


# ─── Helper Functions ──────────────────────────────────────────

def _to_numpy(image) -> np.ndarray:
    """
    Convert various image formats to numpy array (RGB).

    Args:
        image: PIL.Image, numpy array (BGR/RGB), or file path string.

    Returns:
        numpy array in RGB format.
    """
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Cannot read image: {image}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))

    if isinstance(image, np.ndarray):
        # OpenCV uses BGR, convert to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    raise ValueError(f"Unsupported image type: {type(image)}")
