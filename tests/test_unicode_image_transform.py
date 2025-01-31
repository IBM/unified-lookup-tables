"""Tests for ULT text transforms."""

import cv2

from ult.configuration import UnicodeJPEGTransformParameters
from ult.transforms import UnicodeJPEGTransform

# Sample images from cifar10.
IMAGE_PATH = "assets/img1.png"


def test_initialize() -> None:
    """Function to test initialize."""

    unicoder = UnicodeJPEGTransform()
    assert unicoder is not None


def test_unicode_jpeg_transform() -> None:
    """Tests if a set of strings is properly encoded/decoded"""
    parameters = UnicodeJPEGTransformParameters()
    unicoder = UnicodeJPEGTransform(configuration=parameters)

    image = cv2.imread(IMAGE_PATH)
    unicoder.add_multiple_instances([image])

    dimensions = image.shape

    img_encoded = unicoder.encode(image)
    reconstructed_signal = unicoder.decode(text=img_encoded, dimensions=dimensions)

    assert image.shape == reconstructed_signal.shape
