"""Unicode Transforms Parameters."""

from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel


class Occurrences(BaseModel):
    """Occurrences class
    mapping attribute is ex {"1.23": 4}, that is element 1.23 appeared 4 times
    key_type attribute is the real type of mapping keys, this case "float"

    """

    mapping: Dict[str, int]
    key_type: str


class UnicodeTransformParameters(BaseModel):
    """
    Base model for Unicode Parameters
    """

    compression_method: str = "No compression Transform"
    patch_size: int = 8
    offset_value: int = 33
    default_separator: str = " "  # \u0020 for space
    occurrences_path: Optional[Path] = None


class UnicodeTextBWRLETransformParameters(UnicodeTransformParameters):
    """
    Base model for Unicode Parameters
    """

    compression_method: str = "Burrows Wheeler + Run Length Encoding Text Transform"
    default_separator: str = "\u001e"  # Reserved character to avoid the space


class UnicodeJPEGTransformParameters(UnicodeTransformParameters):
    """
    Base model for Unicode Parameters for a JPEG-like image Transform
    """

    compression_method: str = "JPEG-like"
    max_coefficients: int = 3
    dct_precision: int = 1

class UnicodeSeriesCompansionTransformParameters(UnicodeTransformParameters):
    """
    Base model for Unicode Parameters for a Compansion only Series Transform
    """

    compression_method: str = "Compansion"
    companding_max: float = 2**16
    mu: float = 255.0
    float_precision: int = 8