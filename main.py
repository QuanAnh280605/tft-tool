import io

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse

from champion_matcher import ChampionMatcher

app = FastAPI(
    title="TFT Tool API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Initialize champion matcher (loads reference images once at startup)
matcher = ChampionMatcher()


# ─── Shop Detection Config ─────────────────────────────────────
# Shop position ratios (relative to screen size)
SHOP_Y_START_RATIO = 0.86
SHOP_Y_END_RATIO = 0.99
SHOP_X_START_RATIO = 0.245
SHOP_X_END_RATIO = 0.773
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
            
            if champ_name == "empty":
                label = "Empty"
                color = (100, 100, 100)  # Gray for empty
            else:
                label = f"{champ_name} ({conf:.2f})"
                color = (0, 165, 255)  # Orange for champion

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
    # Read uploaded image
    contents = await file.read()
    img_array = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Cannot decode image. Please upload a valid image file."}

    # 1. Detect shop region to get coordinates
    h, w = img.shape[:2]
    x1, y1, x2, y2 = detect_shop_region(img)
    slot_w = (x2 - x1) // NUM_SLOTS

    # Build slot coordinates for matching
    slots = []
    for i in range(NUM_SLOTS):
        slots.append({
            "slot": i + 1,
            "x1": x1 + i * slot_w,
            "y1": y1,
            "x2": x1 + (i + 1) * slot_w,
            "y2": y2,
        })

    # 2. Match champions
    matches = matcher.match_all_slots(img, slots)

    # 3. Draw bounding boxes with champion names
    annotated = draw_shop_bounding_boxes(img, matches)

    # Encode result to PNG
    _, buffer = cv2.imencode(".png", annotated)
    img_bytes = io.BytesIO(buffer.tobytes())

    return StreamingResponse(
        img_bytes,
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=detected_shop.png"},
    )


@app.post("/detect-shop/json", summary="Get shop region coordinates as JSON")
async def detect_shop_json(file: UploadFile = File(...)):
    """
    Upload a TFT screenshot and receive the shop region
    coordinates and slot positions as JSON.
    """
    contents = await file.read()
    
    img_array = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Cannot decode image. Please upload a valid image file."}

    h, w = img.shape[:2]
    x1, y1, x2, y2 = detect_shop_region(img)
    slot_w = (x2 - x1) // NUM_SLOTS

    slots = []
    for i in range(NUM_SLOTS):
        slots.append({
            "slot": i + 1,
            "x1": x1 + i * slot_w,
            "y1": y1,
            "x2": x1 + (i + 1) * slot_w,
            "y2": y2,
        })

    return {
        "image_size": {"width": w, "height": h},
        "shop_region": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "slots": slots,
    }
