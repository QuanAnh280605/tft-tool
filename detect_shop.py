"""
Script to detect and crop the TFT shop region and individual champion slots
from a TFT screenshot.

Improvements:
- Better edge detection for shop boundaries
- Smarter slot detection with padding handling
- More robust to different resolutions
"""

import cv2
import numpy as np
import os

BASE_IMG_PATH = "img/tftbase.png"
OUTPUT_DIR = "img/output"


def analyze_shop_region(img):
    """
    Analyze the TFT screenshot to find shop region.
    The shop is always at the bottom of the screen.
    
    Improved: Uses edge detection to find actual shop boundaries.
    """
    h, w = img.shape[:2]
    print(f"📐 Image size: {w}x{h}")

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Focus on bottom portion where shop is located
    bottom_region_y = int(h * 0.75)
    bottom_region = gray[bottom_region_y:, :]
    
    # Detect edges
    edges = cv2.Canny(bottom_region, 50, 150)
    
    # Find horizontal lines (shop has strong horizontal line at top)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                            minLineLength=w//4, maxLineGap=20)
    
    # Try to find the shop's top edge
    shop_y_start = int(h * 0.86)  # Default fallback
    if lines is not None:
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if line is roughly horizontal and long enough
            if abs(y2 - y1) < 10 and abs(x2 - x1) > w // 4:
                horizontal_lines.append(y1 + bottom_region_y)
        
        if horizontal_lines:
            # Use the topmost horizontal line as shop boundary
            detected_y = min(horizontal_lines)
            if h * 0.7 < detected_y < h * 0.92:  # Sanity check
                shop_y_start = detected_y
                print(f"🔍 Detected shop edge at y={shop_y_start}")
    
    # Calculate shop boundaries
    shop_y_end = int(h * 0.98)
    shop_x_start = int(w * 0.24)
    shop_x_end = int(w * 0.76)

    shop_region = img[shop_y_start:shop_y_end, shop_x_start:shop_x_end]
    print(f"🛒 Shop region: y=[{shop_y_start}:{shop_y_end}], x=[{shop_x_start}:{shop_x_end}]")
    print(f"   Shop size: {shop_region.shape[1]}x{shop_region.shape[0]}")

    return shop_region, (shop_x_start, shop_y_start, shop_x_end, shop_y_end)


def crop_individual_slots(shop_region, num_slots=5):
    """
    Split the shop region into individual champion slots.
    TFT shop always has 5 champion slots evenly spaced.
    
    Improved: 
    - Detects actual card boundaries using contours
    - Falls back to smart division with padding if detection fails
    - Handles gaps between cards properly
    
    Returns:
        List of tuples: (slot_image, (x, y, w, h)) 
        where x,y are relative to shop_region
    """
    h, w = shop_region.shape[:2]
    
    # --- Try contour-based detection first ---
    gray = cv2.cvtColor(shop_region, cv2.COLOR_BGR2GRAY)
    
    # Use adaptive threshold to find card boundaries
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size and aspect ratio (champion cards have specific shape)
    card_candidates = []
    min_area = (w * h) // 50  # At least 2% of shop area
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        
        if area < min_area:
            continue
        
        # Champion cards are taller than wide (portrait orientation)
        aspect = cw / ch if ch > 0 else 0
        if 0.5 < aspect < 1.0:  # Typical card aspect ratio
            card_candidates.append((x, y, cw, ch))
    
    # Sort by x-coordinate (left to right)
    card_candidates.sort(key=lambda c: c[0])
    
    slots = []
    
    # If we detected exactly 5 cards, use them
    if len(card_candidates) == num_slots:
        print(f"✅ Detected {len(card_candidates)} cards using contours")
        for i, (x, y, cw, ch) in enumerate(card_candidates):
            slot = shop_region[y:y+ch, x:x+cw]
            slots.append((slot, (x, y, cw, ch)))
            print(f"   Slot {i+1}: x=[{x}:{x+cw}], y=[{y}:{y+ch}], size={cw}x{ch}")
    else:
        # Fallback: Smart division with padding detection
        print(f"⚠️  Contour detection found {len(card_candidates)} cards, using division method")
        
        # Detect padding by analyzing the shop region
        # Look for vertical gaps (dark columns between cards)
        gray_normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        vertical_profile = np.mean(gray_normalized, axis=0)  # Average brightness per column
        
        # Find dark regions (likely padding between cards)
        threshold = np.percentile(vertical_profile, 25)  # Bottom 25% brightness
        is_dark = vertical_profile < threshold
        
        # Estimate padding by finding consistent dark regions
        padding = 0
        dark_regions = []
        in_dark = False
        dark_start = 0
        
        for i, dark in enumerate(is_dark):
            if dark and not in_dark:
                dark_start = i
                in_dark = True
            elif not dark and in_dark:
                dark_regions.append(i - dark_start)
                in_dark = False
        
        if dark_regions:
            # Use median dark region width as padding estimate
            padding = int(np.median(dark_regions))
            padding = min(padding, w // 30)  # Cap at reasonable value
            print(f"   Detected padding: ~{padding}px")
        else:
            # Default small padding
            padding = max(5, w // 100)
            print(f"   Using default padding: {padding}px")
        
        # Calculate slot width with padding
        total_padding = padding * (num_slots + 1)
        usable_width = w - total_padding
        slot_width = usable_width // num_slots
        
        for i in range(num_slots):
            x_start = padding + i * (slot_width + padding)
            x_end = x_start + slot_width
            slot = shop_region[:, x_start:x_end]
            h_slot = shop_region.shape[0]
            slots.append((slot, (x_start, 0, slot_width, h_slot)))
            print(f"   Slot {i+1}: x=[{x_start}:{x_end}], size={slot.shape[1]}x{slot.shape[0]}")

    return slots


def draw_annotations(img, shop_bbox, num_slots=5):
    """
    Draw bounding boxes on the original image to visualize detected regions.
    
    Improved: Better colors and labels for visibility.
    """
    annotated = img.copy()
    x1, y1, x2, y2 = shop_bbox

    # Draw shop bounding box (green)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(annotated, "SHOP", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Get actual slot positions from shop region
    shop_region = img[y1:y2, x1:x2]
    slots = crop_individual_slots(shop_region, num_slots)
    
    # Calculate slot positions based on actual slots
    shop_w = x2 - x1
    shop_h = y2 - y1
    
    # Use different colors for each slot for better visibility
    colors = [
        (0, 255, 255),   # Yellow
        (255, 255, 0),   # Cyan
        (255, 0, 255),   # Magenta
        (0, 165, 255),   # Orange
        (255, 0, 0),     # Blue
    ]
    
    # For even division fallback
    if len(slots) == num_slots:
        current_x = x1
        current_x = x1
        for i, (slot_img, (sx, sy, sw, sh)) in enumerate(slots):
            slot_w = sw
            # Convert relative slot coords to absolute
            sx1 = x1 + sx
            sy1 = y1 + sy
            sx2 = sx1 + sw
            sy2 = sy1 + sh
            
            color = colors[i % len(colors)]
            cv2.rectangle(annotated, (sx1, sy1), (sx2, sy2), color, 2)
            
            # Add label with background
            label = f"Slot {i+1}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated, 
                         (sx1, sy1 - label_size[1] - 10),
                         (sx1 + label_size[0] + 5, sy1),
                         color, -1)
            cv2.putText(annotated, label, (sx1 + 2, sy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    else:
        # Fallback to simple division
        slot_w = shop_w // num_slots
        for i in range(num_slots):
            sx1 = x1 + i * slot_w
            sx2 = x1 + (i + 1) * slot_w
            color = colors[i % len(colors)]
            cv2.rectangle(annotated, (sx1, y1), (sx2, y2), color, 2)
            cv2.putText(annotated, f"Slot {i+1}", (sx1 + 5, y1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return annotated


def main():
    # Load image
    img = cv2.imread(BASE_IMG_PATH)
    if img is None:
        print(f"❌ Cannot load image: {BASE_IMG_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Find and crop shop region
    print("\n=== 🔍 Detecting Shop Region ===")
    shop_region, shop_bbox = analyze_shop_region(img)
    cv2.imwrite(f"{OUTPUT_DIR}/shop_region.png", shop_region)
    print(f"✅ Saved: {OUTPUT_DIR}/shop_region.png")

    # Step 2: Crop individual slots
    print("\n=== ✂️  Cropping Individual Slots ===")
    slots = crop_individual_slots(shop_region)
    for i, (slot, _) in enumerate(slots):
        cv2.imwrite(f"{OUTPUT_DIR}/slot_{i+1}.png", slot)
    print(f"✅ Saved {len(slots)} slot images")

    # Step 3: Draw annotated image
    print("\n=== 🎨 Drawing Annotations ===")
    annotated = draw_annotations(img, shop_bbox)
    cv2.imwrite(f"{OUTPUT_DIR}/annotated.png", annotated)
    print(f"✅ Saved: {OUTPUT_DIR}/annotated.png")

    print("\n=== 📊 Summary ===")
    print(f"Original image:  {img.shape[1]}x{img.shape[0]}")
    print(f"Shop region:     {shop_region.shape[1]}x{shop_region.shape[0]}")
    if slots:
        avg_w = sum(s[0].shape[1] for s in slots) // len(slots)
        avg_h = sum(s[0].shape[0] for s in slots) // len(slots)
        print(f"Average slot:    {avg_w}x{avg_h}")
    print(f"Slots detected:  {len(slots)}/5")
    print(f"Output dir:      {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()