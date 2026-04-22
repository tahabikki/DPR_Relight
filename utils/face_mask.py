#!/usr/bin/env python3
"""
Face mask utility - ELLIPSE VERSION

Key fixes:
1. Uses ellipse instead of rectangle (follows head shape)
2. Stronger feather (~6% of image) to hide edges
3. Returns empty mask when no face (keeps original)
"""

import numpy as np
import cv2


FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def create_face_mask_opencv(
    image_bgr: np.ndarray,
    with_neck: bool = True,
    dilation: int = 20
) -> np.ndarray:
    """
    Create soft face+neck mask (ellipse-shaped, feathered).
    
    Returns:
        mask: [H, W] uint8 mask (0-255), NOT binary - already feathered
    """
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    if face_cascade.empty():
        print("[Warning] Could not load face cascade")
        return np.ones((h, w), dtype=np.uint8) * 255

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        print("[Warning] No face detected - returning empty mask")
        return mask  # Empty = keep original

    print(f"[Face] Detected {len(faces)} face(s)")

    for (x, y, fw, fh) in faces:
        # Ellipse center: middle of face box, slightly lowered
        cx = x + fw // 2
        cy = y + fh // 2 + int(fh * 0.05)

        # Ellipse axes: wider than face box
        ax = int(fw * 0.60)
        ay = int(fh * 0.80)

        cv2.ellipse(
            mask,
            center=(cx, cy),
            axes=(ax, ay),
            angle=0,
            startAngle=0,
            endAngle=360,
            color=255,
            thickness=-1,
        )

    # Neck: ellipse below face
    if with_neck and len(faces) > 0:
        face_bottom = max(y + fh for (x, y, fw, fh) in faces)
        x_any, _, fw_any, _ = max(faces, key=lambda f: f[2] * f[3])
        face_center_x = x_any + fw_any // 2

        neck_cy = face_bottom + int(h * 0.06)
        neck_ax = int(fw_any * 0.35)
        neck_ay = int(h * 0.09)

        if neck_cy - neck_ay < h:
            cv2.ellipse(
                mask,
                center=(face_center_x, neck_cy),
                axes=(neck_ax, neck_ay),
                angle=0, startAngle=0, endAngle=360,
                color=255, thickness=-1,
            )

    # Dilation
    if dilation > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
        mask = cv2.dilate(mask, kernel, iterations=1)

    # STRONG feather - ~6% of image
    k = max(41, (min(h, w) // 18) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)

    return mask


# Alias for compatibility
create_skin_mask = create_face_mask_opencv


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="mask_visualization.png")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    mask = create_face_mask_opencv(img)
    cv2.imwrite(args.output, mask)
    print(f"Saved {args.output}")