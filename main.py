import io
import logging
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse

import match_service
import ocr_service

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="TFT Tool API",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.on_event("startup")
async def startup_event():
    """Pre-load models at startup to avoid first-request delay."""
    ocr_service.get_reader()
    match_service.get_champion_db()



# ─── Shop Detection Config ─────────────────────────────────────
# Shop position ratios (relative to screen size)
SHOP_Y_START_RATIO = 0.86
SHOP_Y_END_RATIO = 0.99
SHOP_X_START_RATIO = 0.267
SHOP_X_END_RATIO = 0.752
NUM_SLOTS = 5


def detect_shop_region(img: np.ndarray) -> tuple:
    """Detect the shop region coordinates from a TFT screenshot."""
    h, w = img.shape[:2]
    y1 = int(h * SHOP_Y_START_RATIO)
    y2 = int(h * SHOP_Y_END_RATIO)
    x1 = int(w * SHOP_X_START_RATIO)
    x2 = int(w * SHOP_X_END_RATIO)
    return x1, y1, x2, y2


def draw_shop_bounding_boxes(img: np.ndarray, matches: list[dict] = None) -> np.ndarray:
    """
    Draw bounding boxes around shop region and individual slots.
    If matches are provided, draw champion names instead of slot numbers.
    """
    annotated = img.copy()
    x1, y1, x2, y2 = detect_shop_region(img)

    # Draw shop bounding box (green, thick)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(
        annotated, "SHOP", (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
    )

    # Draw individual slot boxes
    slot_w = (x2 - x1) // NUM_SLOTS
    for i in range(NUM_SLOTS):
        sx1 = x1 + i * slot_w
        sx2 = x1 + (i + 1) * slot_w
        
        # Determine label and color
        label = f"Slot {i+1}"
        color = (0, 255, 255)  # Yellow default
        
        if matches and i < len(matches):
            match = matches[i]
            champ_name = match.get("champion", "unknown")
            conf = match.get("confidence", 0.0)
            
            if champ_name:
                label = f"{champ_name} ({conf:.2f})"
                color = (0, 165, 255)
            else:
                label = ""  # Or f"{conf:.2f}" if you want to see confidence
                color = (100, 100, 100)

        cv2.rectangle(annotated, (sx1, y1), (sx2, y2), color, 2)
        
        # Draw text with background for better visibility
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (sx1, y1 - 20), (sx1 + w, y1), color, -1)
        cv2.putText(
            annotated, label, (sx1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
        )

    return annotated


# ─── Routes ─────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "TFT Tool API is running"}


@app.post("/detect-shop", summary="Detect shop and draw champion names")
async def detect_shop(file: UploadFile = File(...)):
    """
    Upload a TFT screenshot and receive the image back
    with bounding boxes and identified champion names.
    """
    t_start = time.perf_counter()

    # Read uploaded image
    contents = await file.read()
    img_array = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Cannot decode image. Please upload a valid image file."}

    # 1. Detect shop region to get coordinates
    h, w = img.shape[:2]
    x1, y1, x2, y2 = detect_shop_region(img)
    shop_region = img[y1:y2, x1:x2]

    t_ocr_start = time.perf_counter()

    # 2. OCR champion names (single-call optimized)
    ocr_results = ocr_service.recognize_name_strip(shop_region, NUM_SLOTS)

    t_ocr_end = time.perf_counter()

    # 3. Build matches from OCR results
    matches = []
    for item in ocr_results:
        text = item["text"]
        matches.append({
            "champion": text,
            "confidence": item["confidence"],
        })

    # 4. Draw bounding boxes with champion names
    annotated = draw_shop_bounding_boxes(img, matches)

    # Encode result to PNG
    success, encoded = cv2.imencode(".png", annotated)
    if not success:
        return {"error": "Failed to encode result image."}

    t_total = time.perf_counter() - t_start
    t_ocr = t_ocr_end - t_ocr_start
    logging.info(
        "⏱️ detect-shop | OCR: %.0fms | Total: %.0fms | Champions: %s",
        t_ocr * 1000, t_total * 1000,
        [m["champion"] for m in matches],
    )

    img_bytes = io.BytesIO(encoded.tobytes())

    return StreamingResponse(
        img_bytes,
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=detected_shop.png"},
    )


@app.post("/detect-shop/json", summary="Get shop region coordinates as JSON")
async def detect_shop_json(file: UploadFile = File(...)):
    """
    Upload a TFT screenshot and receive the shop region
    coordinates, slot positions, and OCR text as JSON.
    Uses EasyOCR (slower but potentially more accurate for some fonts).
    """
    t_start = time.perf_counter()

    contents = await file.read()

    img_array = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Cannot decode image. Please upload a valid image file."}

    h, w = img.shape[:2]
    x1, y1, x2, y2 = detect_shop_region(img)

    t_ocr_start = time.perf_counter()

    # OCR champion names (single-call optimized)
    shop_region = img[y1:y2, x1:x2]
    ocr_results = ocr_service.recognize_name_strip(shop_region, NUM_SLOTS)

    t_ocr_end = time.perf_counter()

    # Build slots with absolute coordinates
    slots = []
    for item in ocr_results:
        bbox = item["bbox"]
        slots.append({
            "slot": item["slot"],
            "text": item["text"],
            "confidence": item["confidence"],
            "x1": x1 + bbox["x1"],
            "y1": y1 + bbox["y1"],
            "x2": x1 + bbox["x2"],
            "y2": y1 + bbox["y2"],
        })

    t_total = time.perf_counter() - t_start
    t_ocr = t_ocr_end - t_ocr_start

    logging.info(
        "⏱️ detect-shop/json (EasyOCR) | OCR: %.0fms | Total: %.0fms | Champions: %s",
        t_ocr * 1000, t_total * 1000,
        [s["text"] for s in slots],
    )

    return {
        "engine": "easyocr",
        "timing": {
            "ocr_ms": round(t_ocr * 1000, 1),
            "total_ms": round(t_total * 1000, 1),
        },
        "image_size": {"width": w, "height": h},
        "shop_region": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "slots": slots,
    }


@app.post("/ocr", summary="OCR any image to text")
async def ocr_image(file: UploadFile = File(...)):
    """
    Upload any image and receive the recognized text.
    Uses EasyOCR model.
    """
    contents = await file.read()
    img_array = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Cannot decode image. Please upload a valid image file."}

    text = ocr_service.recognize_text(img)

    return {
        "text": text,
        "model": "easyocr",
    }


@app.post("/detect-shop/tesseract", summary="Detect shop with Tesseract OCR (fast)")
async def detect_shop_tesseract(file: UploadFile = File(...)):
    """
    Upload a TFT screenshot and receive champion names using Tesseract OCR.
    Faster alternative to EasyOCR (~5-10x speed improvement).
    Returns JSON with timing info for benchmarking.
    """
    t_start = time.perf_counter()

    contents = await file.read()
    img_array = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Cannot decode image. Please upload a valid image file."}

    # 1. Detect shop region
    h, w = img.shape[:2]
    x1, y1, x2, y2 = detect_shop_region(img)
    shop_region = img[y1:y2, x1:x2]

    t_ocr_start = time.perf_counter()

    # 2. OCR with Tesseract
    ocr_results = ocr_service.recognize_name_strip_tesseract(shop_region, NUM_SLOTS)

    t_ocr_end = time.perf_counter()

    # 3. Build response
    slots = []
    for item in ocr_results:
        bbox = item["bbox"]
        slots.append({
            "slot": item["slot"],
            "text": item["text"],
            "confidence": item["confidence"],
            "x1": x1 + bbox["x1"],
            "y1": y1 + bbox["y1"],
            "x2": x1 + bbox["x2"],
            "y2": y1 + bbox["y2"],
        })

    t_total = time.perf_counter() - t_start
    t_ocr = t_ocr_end - t_ocr_start

    logging.info(
        "⏱️ detect-shop/tesseract | OCR: %.0fms | Total: %.0fms | Champions: %s",
        t_ocr * 1000, t_total * 1000,
        [s["text"] for s in slots],
    )

    return {
        "engine": "tesseract",
        "timing": {
            "ocr_ms": round(t_ocr * 1000, 1),
            "total_ms": round(t_total * 1000, 1),
        },
        "image_size": {"width": w, "height": h},
        "shop_region": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "slots": slots,
    }


@app.post("/detect-shop/match", summary="Detect shop with portrait matching (ultra-fast)")
async def detect_shop_match(file: UploadFile = File(...)):
    """
    Upload a TFT screenshot and identify champions using
    image hash matching (~1-5ms vs ~400-600ms for OCR).
    Returns annotated image with bounding boxes.
    """
    t_start = time.perf_counter()

    contents = await file.read()
    img_array = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Cannot decode image. Please upload a valid image file."}

    # 1. Detect shop region
    h, w = img.shape[:2]
    x1, y1, x2, y2 = detect_shop_region(img)
    shop_region = img[y1:y2, x1:x2]

    t_match_start = time.perf_counter()

    # 2. Match champions using portrait hashing
    match_results = match_service.match_shop_slots(shop_region, NUM_SLOTS)

    t_match_end = time.perf_counter()

    # 3. Build matches list compatible with draw_shop_bounding_boxes
    matches = []
    for item in match_results:
        champ = item["champion"] or item.get("best_guess", "")
        label = champ if item["champion"] else f"?{champ}"
        matches.append({
            "champion": label,
            "confidence": item["confidence"],
            "distance": item["distance"],
        })

    # 4. Draw bounding boxes with champion names
    annotated = draw_shop_bounding_boxes(img, matches)

    # Encode result to PNG
    success, encoded = cv2.imencode(".png", annotated)
    if not success:
        return {"error": "Failed to encode result image."}

    t_total = time.perf_counter() - t_start
    t_match = t_match_end - t_match_start
    logging.info(
        "⏱️ detect-shop/match | Match: %.1fms | Total: %.1fms | Champions: %s",
        t_match * 1000, t_total * 1000,
        [m["champion"] for m in matches],
    )

    img_bytes = io.BytesIO(encoded.tobytes())

    return StreamingResponse(
        img_bytes,
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=detected_match.png"},
    )
