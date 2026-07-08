"""Arabic handwritten text recognition (CRNN-CTC) trained on the KHATT dataset."""

__version__ = "0.1.0"

from .pipeline import ArabicOCR

__all__ = ["ArabicOCR", "__version__"]
