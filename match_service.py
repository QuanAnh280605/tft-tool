"""
Champion Portrait Matching Service

Uses perceptual hashing (pHash) to match TFT shop slot portraits
against a pre-built database of known champion card images.

~1-5ms for 5 slots vs ~400-600ms for OCR.
"""

import logging
import os
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─── Singleton Champion Database ───────────────────────────────
_champion_db = None

# Portrait crop ratios (relative to full card image)
# For reference card images (img/*.png):
#   - Top ~30%: trait icons overlaid on portrait
#   - Middle ~40%: main champion portrait
#   - Bottom ~20-25%: name + cost
# We want the middle portrait area for matching
CARD_PORTRAIT_TOP = 0.15
CARD_PORTRAIT_BOTTOM = 0.70
CARD_PORTRAIT_LEFT = 0.08
CARD_PORTRAIT_RIGHT = 0.92

# For shop slot images cropped from screenshot:
# Same general layout but might differ slightly
SLOT_PORTRAIT_TOP = 0.10
SLOT_PORTRAIT_BOTTOM = 0.65
SLOT_PORTRAIT_LEFT = 0.08
SLOT_PORTRAIT_RIGHT = 0.92

# Matching threshold (Hamming distance)
# Lower = more similar. 64-bit hash = max distance 64
MATCH_THRESHOLD = 22  # Relaxed threshold for game screenshot vs reference images
HASH_IMG_SIZE = 32     # Resize to 32x32 before DCT
HASH_SIZE = 8          # Take top-left 8x8 of DCT = 64-bit hash


def _compute_phash(image: np.ndarray) -> int:
    """
    Compute perceptual hash (pHash) manually using DCT.

    Steps:
    1. Convert to grayscale
    2. Resize to 32x32
    3. Apply DCT
    4. Take top-left 8x8 block (low frequencies)
    5. Compute median and build 64-bit hash
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    resized = cv2.resize(
        gray, (HASH_IMG_SIZE, HASH_IMG_SIZE),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)

    dct_result = cv2.dct(resized)
    dct_low = dct_result[:HASH_SIZE, :HASH_SIZE]

    median_val = np.median(dct_low)

    hash_bits = (dct_low > median_val).flatten()
    hash_val = 0
    for bit in hash_bits:
        hash_val = (hash_val << 1) | int(bit)

    return hash_val


def _hamming_distance(hash1: int, hash2: int) -> int:
    """Compute Hamming distance between two 64-bit hashes."""
    xor = hash1 ^ hash2
    return bin(xor).count('1')


def _crop_card_portrait(card_img: np.ndarray) -> np.ndarray:
    """Crop portrait from a reference card image (img/*.png)."""
    h, w = card_img.shape[:2]
    y1 = int(h * CARD_PORTRAIT_TOP)
    y2 = int(h * CARD_PORTRAIT_BOTTOM)
    x1 = int(w * CARD_PORTRAIT_LEFT)
    x2 = int(w * CARD_PORTRAIT_RIGHT)
    return card_img[y1:y2, x1:x2]


def _crop_slot_portrait(slot_img: np.ndarray) -> np.ndarray:
    """Crop portrait from a shop slot image (from screenshot)."""
    h, w = slot_img.shape[:2]
    y1 = int(h * SLOT_PORTRAIT_TOP)
    y2 = int(h * SLOT_PORTRAIT_BOTTOM)
    x1 = int(w * SLOT_PORTRAIT_LEFT)
    x2 = int(w * SLOT_PORTRAIT_RIGHT)
    return slot_img[y1:y2, x1:x2]


def _load_champion_db(img_dir: str) -> dict:
    """
    Load all champion card images, crop portraits, compute hashes.

    Returns:
        Dict mapping champion name -> {'hash': int, 'portrait_size': tuple}
    """
    db = {}
    skip_files = {"tftbase.png"}

    for filename in os.listdir(img_dir):
        filepath = os.path.join(img_dir, filename)

        if os.path.isdir(filepath):
            continue
        if filename in skip_files:
            continue
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img = cv2.imread(filepath)
        if img is None:
            logger.warning("Cannot load image: %s", filepath)
            continue

        name = os.path.splitext(filename)[0]

        portrait = _crop_card_portrait(img)
        phash = _compute_phash(portrait)

        db[name] = {
            "hash": phash,
            "portrait_size": portrait.shape[:2],
        }

    logger.info(
        "✅ Loaded champion database: %d champions from '%s'",
        len(db), img_dir,
    )
    return db


def get_champion_db(img_dir: str = "img") -> dict:
    """Get or initialize the champion hash database (singleton)."""
    global _champion_db
    if _champion_db is None:
        t_start = time.perf_counter()
        _champion_db = _load_champion_db(img_dir)
        t_elapsed = time.perf_counter() - t_start
        logger.info(
            "⏱️ Champion DB loaded in %.0fms (%d champions)",
            t_elapsed * 1000, len(_champion_db),
        )
    return _champion_db


def match_slot_portrait(
    slot_img: np.ndarray,
    db: dict,
) -> dict:
    """
    Match a single shop slot image against the champion database.

    Returns:
        Dict with 'champion', 'distance', 'confidence' keys.
    """
    portrait = _crop_slot_portrait(slot_img)
    slot_hash = _compute_phash(portrait)

    # Find best match
    best_name = ""
    best_distance = 999
    top_matches = []

    for name, data in db.items():
        distance = _hamming_distance(slot_hash, data["hash"])
        top_matches.append((name, distance))
        if distance < best_distance:
            best_distance = distance
            best_name = name

    # Log top 3 matches for debugging
    top_matches.sort(key=lambda x: x[1])
    top3 = top_matches[:3]
    logger.debug(
        "🔍 Slot hash match top3: %s",
        [(n, d) for n, d in top3],
    )

    # Convert distance to confidence (0.0 - 1.0)
    confidence = max(0.0, 1.0 - (best_distance / 64.0))

    if best_distance > MATCH_THRESHOLD:
        logger.info(
            "❌ No match (best: '%s' dist=%d > threshold=%d)",
            best_name, best_distance, MATCH_THRESHOLD,
        )
        return {
            "champion": "",
            "distance": int(best_distance),
            "confidence": 0.0,
            "best_guess": best_name,
        }

    return {
        "champion": best_name,
        "distance": int(best_distance),
        "confidence": round(confidence, 4),
        "best_guess": best_name,
    }


def match_shop_slots(
    shop_region: np.ndarray,
    num_slots: int = 5,
) -> list[dict]:
    """
    Match all champion slots in the shop region.

    Returns:
        List of dicts with 'slot', 'champion', 'distance', 'confidence' keys.
    """
    db = get_champion_db()
    shop_h, shop_w = shop_region.shape[:2]
    slot_w = shop_w // num_slots

    results = []
    for i in range(num_slots):
        sx1 = i * slot_w
        sx2 = (i + 1) * slot_w if i < num_slots - 1 else shop_w
        slot_img = shop_region[:, sx1:sx2]

        match = match_slot_portrait(slot_img, db)
        match["slot"] = i + 1
        match["bbox"] = {
            "x1": sx1, "y1": 0, "x2": sx2, "y2": shop_h,
        }
        results.append(match)

    return results
