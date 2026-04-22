#!/usr/bin/env python3
"""
Skin mask utility using color-based detection

Detects skin regions using YCrCb color space (better than HSV for skin).
Excludes hair, eyes, mouth, and background.
"""

import numpy as np
import cv2


def detect_skin_ycrcb(image_bgr: np.ndarray) -> np.ndarray:
    """
    Detect skin regions using YCrCb color space.
    """
    # Convert to YCrCb
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)

    # Expanded skin color range for diverse skin tones
    # Includes darker skin tones
    lower = np.array([0, 128, 65])
    upper = np.array([255, 180, 145])

    mask = cv2.inRange(ycrcb, lower, upper)

    return mask


def detect_skin_hsv(image_bgr: np.ndarray) -> np.ndarray:
    """
    Detect skin regions using HSV color space.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Expanded skin range for better detection
    lower = np.array([0, 15, 50])
    upper = np.array([20, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    return mask


def detect_skin_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """
    Detect skin using RGB rules.
    """
    b, g, r = cv2.split(image_bgr.astype(np.float32))
    
    # RGB skin detection rules
    mask = (r > 95) & (g > 40) & (b > 20) & \
           ((r - g) > 15) & ((r - b) > 15) & \
           (r > g) & (r > b)
    
    return (mask.astype(np.uint8) * 255)


def create_skin_mask_color(
    image_bgr: np.ndarray,
    use_morphology: bool = True,
    min_skin_area: int = 3000
) -> np.ndarray:
    """
    Create skin mask using color detection.
    
    Args:
        image_bgr: Input BGR image
        use_morphology: Apply morphological cleanup
        min_skin_area: Minimum connected region to keep
    
    Returns:
        Binary mask (255 = skin, 0 = non-skin)
    """
    h, w = image_bgr.shape[:2]

    # Get skin masks from multiple methods
    mask_ycrcb = detect_skin_ycrcb(image_bgr)
    mask_hsv = detect_skin_hsv(image_bgr)
    mask_rgb = detect_skin_rgb(image_bgr)

    # Combine (union for coverage)
    mask = cv2.bitwise_or(mask_ycrcb, mask_hsv)
    mask = cv2.bitwise_or(mask, mask_rgb)

    # Morphological cleanup
    if use_morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find largest connected component (main face region)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create clean mask
    clean_mask = np.zeros((h, w), dtype=np.uint8)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_skin_area:
            cv2.drawContours(clean_mask, [contour], -1, 255, -1)

    # Slight dilation for coverage
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean_mask = cv2.dilate(clean_mask, kernel, iterations=1)

    # STRONG feather - ~6% of image (fixed for smooth edges!)
    k = max(41, (min(h, w) // 18) | 1)
    clean_mask = cv2.GaussianBlur(clean_mask, (k, k), 0)

    return clean_mask


def create_face_mask_color(
    image_bgr: np.ndarray,
    with_neck: bool = True
) -> np.ndarray:
    """
    Create combined mask (face + neck) using color detection only.
    Skip hard rectangles - just use smooth color detection to avoid visible edges.
    """
    h, w = image_bgr.shape[:2]

    # Use only soft color-based skin mask, NO hard face detection rectangles
    skin_mask = create_skin_mask_color(image_bgr)

    # EXTRA strong feathering to eliminate any visible edge artifacts
    # Use very large blur kernel
    k = max(61, (min(h, w) // 15) | 1)
    skin_mask = cv2.GaussianBlur(skin_mask, (k, k), 0)

    return skin_mask


# Alias for compatibility
create_skin_mask = create_face_mask_color


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    mask = create_face_mask_color(img)

    vis = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite("mask_color.png", vis)
    print("Saved mask_color.png")