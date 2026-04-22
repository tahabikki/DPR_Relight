#!/usr/bin/env python3
"""
Face mask utility using OpenCV Haar Cascade

Creates a mask for face + neck skin areas using OpenCV's face detector.
Faster and no extra dependencies.
"""

import numpy as np
import cv2


# Default cascade path (comes with OpenCV)
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def create_face_mask_opencv(
    image_bgr: np.ndarray,
    with_neck: bool = True,
    dilation: int = 20
) -> np.ndarray:
    """
    Create a mask for skin regions (face + neck) using OpenCV.
    
    Args:
        image_bgr: Input image in BGR format
        with_neck: Include neck region
        dilation: Mask dilation amount
    
    Returns:
        mask: [H, W] binary mask (255 = skin, 0 = non-skin)
    """
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    
    if face_cascade.empty():
        print("[Warning] Could not load face cascade")
        return np.ones((h, w), dtype=np.uint8) * 255
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
    
    if len(faces) == 0:
        print("[Warning] No face detected")
        return np.ones((h, w), dtype=np.uint8) * 255
    
    print(f"[Face] Detected {len(faces)} face(s)")
    
    # Process each face
    for (x, y, fw, fh) in faces:
        # Face region
        face_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Add face rectangle with some padding
        pad = int(fw * 0.15)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + fw + pad)
        y2 = min(h, y + fh + int(fh * 1.1))  # Extend slightly below face
        
        cv2.rectangle(face_mask, (x1, y1), (x2, y2), 255, -1)
        
        # Add to mask
        mask = cv2.bitwise_or(mask, face_mask)
        
        print(f"  Face at ({x},{y}), size {fw}x{fh}")
    
    # Add neck region (below face)
    if with_neck and len(faces) > 0:
        # Find lowest face bottom
        face_bottom = max(y + fh for (x, y, fw, fh) in faces)
        
        # Add neck rectangle below face
        neck_top = face_bottom
        neck_bottom = min(h, face_bottom + int(h * 0.15))  # ~15% of image height
        
        if neck_bottom > neck_top:
            neck_left = int(w * 0.35)
            neck_right = int(w * 0.65)
            
            neck_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(neck_mask, (neck_left, neck_top), (neck_right, neck_bottom), 255, -1)
            
            mask = cv2.bitwise_or(mask, neck_mask)
            print(f"  Neck region: ({neck_left},{neck_top}) to ({neck_right},{neck_bottom})")
    
    # Dilate for more coverage
    if dilation > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Smooth edges
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    
    return mask


# Alias for compatibility
create_skin_mask = create_face_mask_opencv


# Ensure cascade is available
def ensure_cascade():
    """Verify OpenCV cascade is available."""
    import os
    cascade_path = cv2.data.haarcascades
    print(f"[Info] OpenCV cascade path: {cascade_path}")
    print(f"[Info] Available files: {os.listdir(cascade_path)[:5]}...")


if __name__ == "__main__":
    ensure_cascade()
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    
    img = cv2.imread(args.image)
    mask = create_face_mask_opencv(img)
    
    vis = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite("mask_visualization.png", vis)
    print("Saved mask_visualization.png")