"""Unified Lookup Tables Data Module."""

from .images.cifar import CifarImageDataLoader
from .images.pass_images import PASSImageDataLoader
from .text.core import WikiText103Loader

DATASETLOADER_REGISTRY = {
    "wikitext": WikiText103Loader,
    "pass": PASSImageDataLoader,
    "cifar10": CifarImageDataLoader,
}
