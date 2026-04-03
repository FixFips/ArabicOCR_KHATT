# src/augment.py
"""
Arabic-safe data augmentation for handwriting OCR.

DESIGN PRINCIPLE: Arabic letters (ba/ta/tha/nun/ya) share identical base
strokes and differ ONLY by dots that are 2-4 pixels tall.  Every transform
here is chosen to preserve dots and cursive connections.

BANNED transforms and why:
  - Erosion:           destroys dots (2-4px) and breaks thin connections
  - Dilation:          merges adjacent dots (ta → ba) and fills inter-letter gaps
  - Elastic distortion: shifts dots away from their base letter
  - Random crop/cutout: can remove dots above/below the text line
  - Heavy rotation (>2°): shifts dot laterally to wrong letter position
"""

import random
import cv2
import numpy as np
from PIL import Image


class ArabicAugment:
    """Apply Arabic-safe augmentations to a binarized grayscale PIL Image."""

    def __init__(self, training: bool = True):
        self.training = training

    def __call__(self, img: Image.Image) -> Image.Image:
        if not self.training:
            return img

        arr = np.array(img, dtype=np.uint8)  # binarized grayscale

        # 1) Affine shear — slants all features equally, dots stay above/below their letter
        if random.random() < 0.4:
            arr = self._shear(arr)

        # 2) Slight rotation — within ±2° dots keep their letter association
        if random.random() < 0.3:
            arr = self._rotate(arr)

        # 3) Kashida (tatweel) stretch — uniquely Arabic horizontal elongation
        #    Stretches connections between letters; dots sit above/below and are unaffected
        if random.random() < 0.3:
            arr = self._kashida_stretch(arr)

        # 4) Vertical baseline shift — simulates off-center writing
        if random.random() < 0.2:
            arr = self._vertical_shift(arr)

        # 5) Brightness/contrast jitter — simulates paper/ink variation
        if random.random() < 0.3:
            arr = self._brightness_jitter(arr)

        # 6) Mild Gaussian noise — scanner/paper noise at safe levels
        if random.random() < 0.2:
            arr = self._gaussian_noise(arr)

        # Re-binarize to maintain clean binary format
        arr = (arr > 128).astype(np.uint8) * 255

        return Image.fromarray(arr, mode="L")

    @staticmethod
    def _shear(arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape[:2]
        shear_x = random.uniform(-0.12, 0.12)
        M = np.array([[1, shear_x, 0], [0, 1, 0]], dtype=np.float32)
        # Shift to keep content centered
        M[0, 2] = -shear_x * h / 2
        return cv2.warpAffine(
            arr, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

    @staticmethod
    def _rotate(arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape[:2]
        angle = random.uniform(-2.0, 2.0)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(
            arr, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

    @staticmethod
    def _kashida_stretch(arr: np.ndarray) -> np.ndarray:
        """
        Simulate Arabic kashida (tatweel) by horizontally stretching
        a random column range.  This stretches the horizontal connections
        between letters — dots sit above/below and are spatially unaffected.
        """
        h, w = arr.shape[:2]
        if w < 20:
            return arr

        # Pick a random stretch region (20-40% of width)
        region_w = random.randint(w // 5, 2 * w // 5)
        start = random.randint(0, w - region_w)
        end = start + region_w

        stretch_factor = random.uniform(1.0, 1.15)
        new_region_w = int(region_w * stretch_factor)

        region = arr[:, start:end]
        stretched = cv2.resize(region, (new_region_w, h), interpolation=cv2.INTER_LINEAR)

        new_w = w - region_w + new_region_w
        result = np.full((h, new_w), 255, dtype=np.uint8)
        result[:, :start] = arr[:, :start]
        result[:, start:start + new_region_w] = stretched
        result[:, start + new_region_w:] = arr[:, end:]

        # Resize back to original width to maintain consistency
        return cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _vertical_shift(arr: np.ndarray) -> np.ndarray:
        shift = random.randint(-2, 2)
        if shift == 0:
            return arr
        M = np.array([[1, 0, 0], [0, 1, shift]], dtype=np.float32)
        return cv2.warpAffine(
            arr, M, (arr.shape[1], arr.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

    @staticmethod
    def _brightness_jitter(arr: np.ndarray) -> np.ndarray:
        factor = random.uniform(0.85, 1.15)
        offset = random.uniform(-10, 10)
        result = arr.astype(np.float32) * factor + offset
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _gaussian_noise(arr: np.ndarray) -> np.ndarray:
        sigma = random.uniform(0, 8)
        noise = np.random.randn(*arr.shape) * sigma
        result = arr.astype(np.float32) + noise
        return np.clip(result, 0, 255).astype(np.uint8)
