"""
PREPROCESSING (Inference Only)
Used during model deployment for RayScan
"""

import cv2
import numpy as np


def zscore_normalize(image, eps=1e-7):
    """
    Apply Z-score normalization
    """
    image = image.astype(np.float32)
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / (std + eps)


def custom_preprocess(img):
    """
    Preprocessing pipeline used during training
    (without augmentation, for inference only)
    """

    # Ensure uint8
    img = img.astype(np.uint8)

    # Convert to grayscale if RGB
    if img.ndim == 3 and img.shape[-1] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.squeeze()

    # Noise reduction
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Convert back to RGB (model expects 3 channels)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # Z-score normalization
    img = zscore_normalize(img)

    return img
