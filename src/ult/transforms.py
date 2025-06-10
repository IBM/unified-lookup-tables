"""Unicode transforms."""

import bisect
import json
import multiprocessing as mp
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, cast

import numpy as np
import skimage
from loguru import logger
from numpy.typing import NDArray
from scipy.fft import dctn, idctn
from skimage.util import view_as_blocks

from .configuration import (
    Occurrences,
    UnicodeJPEGTransformParameters,
    UnicodeTextBWRLETransformParameters,
    UnicodeTransformParameters,
    UnicodeSeriesCompansionTransformParameters,
)

LAST_UNICODE_POINT = 2**24
LAST_UNICODE_CHAR = chr(1114111)  # safe option for some cases chr(149000)
OccurrenceType = float | int | str
OccurrencesSequenceType = List[List[OccurrenceType]]


class UnicodeDataTransform(ABC):
    """Class that implements the unicode transform as a compression method."""

    def __init__(
        self, configuration: UnicodeTransformParameters = UnicodeTransformParameters()
    ):
        """Creates a unicode data transform.

        Args:
            patch_size: patch size for instance compression algorithm. Default to 16.
            offset_value: initial unicode character offset to use for pattern occurrences.
            compression_parameters: optional compression parameters.
        """
        # initialize private attributes
        self._lut: OrderedDict[OccurrenceType, str] = OrderedDict()
        self._lut_key_list: List[OccurrenceType] = list(self._lut.keys())
        self._inverse_lut: Dict[str, OccurrenceType] = dict()
        self._occurrences: OrderedDict[OccurrenceType, int] = OrderedDict()
        # store parameters for compression
        self.patch_size = configuration.patch_size
        self.offset_value = configuration.offset_value
        self.compression_method = configuration.compression_method
        self.default_separator = configuration.default_separator
        # if the configuration contains a path to occurrences, they are loaded and set
        if configuration.occurrences_path is not None:
            with configuration.occurrences_path.open("rt") as fp:
                # keys in json are strings, occurrences objects might not
                occurrences_json = json.load(fp)
                occurrences_object = Occurrences(**occurrences_json)
                # cast function for keys str -> occurrence type
                cast_key = eval(occurrences_object.key_type)
                # occurrences keys are occurrence type not str
                occurrences = OrderedDict(
                    {
                        cast_key(key): value
                        for key, value in occurrences_object.mapping.items()
                    }
                )
                self.occurrences = occurrences

    @property
    def occurrences(self) -> OrderedDict[OccurrenceType, int]:
        """Occurrences counts getter.

        Returns:
            occurrences counts.
        """
        return self._occurrences

    @occurrences.setter
    def occurrences(self, occurrences: OrderedDict[OccurrenceType, int]) -> None:
        """Occurrences setter.

        This ensure look-up tables are rebuilt.

        Args:
            occurrences: occurrences object.
        """
        # we make sure occurrences are sorted from most to least frequent ones.
        self._occurrences = OrderedDict(
            [
                (occurrence, count)
                for occurrence, count in sorted(
                    occurrences.items(), key=lambda item: item[1], reverse=True
                )
            ]
        )
        # build look-up table
        self._lut = OrderedDict()
        for index, key in enumerate(self._occurrences.keys()):
            if (index + self.offset_value) <= LAST_UNICODE_POINT:
                unicode_char = chr(index + self.offset_value)
            else:
                unicode_char = chr(LAST_UNICODE_POINT)

            self._lut[key] = unicode_char.encode("utf-8", errors="replace").decode(
                "utf-8"
            )
        # re-sort LUT and store keys for fast look-ups
        self._lut = OrderedDict(
            [
                (key, value)
                for key, value in sorted(self._lut.items(), key=lambda item: item[0])
            ]
        )
        self._lut_key_list = list(self._lut.keys())
        # build inverse look-up table
        self._inverse_lut = {
            string_representation: occurrence
            for occurrence, string_representation in self._lut.items()
        }
        self._inverse_lut[LAST_UNICODE_CHAR] = self._lut[next(reversed(self._lut))]

    def save_occurrences(self, filepath: Path) -> None:
        """Save occurrences to disk.

        Args:
            filepath: path where to save occurrences.
        """
        with filepath.open("wt") as fp:
            if isinstance(list(self._occurrences.keys())[0], str):
                key_type = "str"
            elif isinstance(list(self._occurrences.keys())[0], int) or isinstance(
                list(self._occurrences.keys())[0], np.integer
            ):
                key_type = "int"
            elif isinstance(list(self._occurrences.keys())[0], float) or isinstance(
                list(self._occurrences.keys())[0], np.floating
            ):
                key_type = "float"
            else:
                key_type = "str"
            mapping = dict(
                [(str(key), value) for key, value in self._occurrences.items()]
            )

            occurrences_object = Occurrences(mapping=mapping, key_type=key_type)
            json.dump(occurrences_object.model_dump(), fp)

    @property
    def lut(self) -> OrderedDict[OccurrenceType, str]:
        """Look-up table getter.

        Returns:
            the look-up table.
        """
        return self._lut

    @property
    def lut_key_list(self) -> List[OccurrenceType]:
        """Look-up table key list getter.

        Returns:
            look-up table key list.
        """
        return self._lut_key_list

    @property
    def inverse_lut(self) -> Dict[str, OccurrenceType]:
        """Inverse look-up table getter.

        Returns:
            inverse look-up table.
        """
        return self._inverse_lut

    @abstractmethod
    def compress(self, instance: Any) -> OccurrencesSequenceType:
        """Compress an instance.

        Args:
            instance: instance to compress.

        Returns:
            the instance compressed as an occurrence sequence.
        """

    @abstractmethod
    def decompress(
        self, sequence: OccurrencesSequenceType, dimensions: Optional[List[int]] = None
    ) -> Any:
        """Decompress a sequence in an instance.

        Args:
            sequence: sequence to decompress.
            dimensions: optional dimensions for decompression. Defaults to None.

        Returns:
            instance decompressed from an occurrence sequence.
        """

    def lut_lookup(self, value: OccurrenceType) -> str:
        """Efficient look-up up in the table.

        Args:
            value: an occurrence value.

        Returns:
            string representation for the occurrence.
        """
        key_index = bisect.bisect_left(self._lut_key_list, value)
        # ensure that it is in range
        key_index = max(0, min(len(self._lut_key_list) - 1, key_index))
        return self._lut[self._lut_key_list[key_index]]

    def encode(self, instance: Any) -> str:
        """Encodes an instance using the look-up table.

        Args:
            instance: instance to encode.

        Returns:
            text representation of the instance.
        """
        sequence = self.compress(instance)
        text = ""
        # NOTE: this should not be needed unless we relax typing
        if sequence is not None:
            text = self.default_separator.join(
                [
                    "".join([self.lut_lookup(value) for value in patch])
                    for patch in sequence
                ]
            )
        return text

    def decode(self, text: str, dimensions: Optional[List[int,]] = None) -> Any:
        """Decodes a text string into an instance using the inverse look-up table.

        Args:
            text: text representation of the instance
            dimensions: dimensions of the instance to decode.

        Returns:
            decoded instance.
        """
        sequence = [
            [
                self._inverse_lut.get(character, self._inverse_lut[LAST_UNICODE_CHAR])
                for character in word
            ]
            for word in text.split(self.default_separator)
        ]

        # NOTE: removes empty lists caused by the default separator.
        # NOTE: empty lists should not be removed to preserve sequence length
        # NOTE: empty lists should be removed otherwise decoding fails.
        cleaned_sequence = [item for item in sequence if item]

        return self.decompress(cleaned_sequence, dimensions)

    def update_occurrences_from_sequence(
        self, sequence: OccurrencesSequenceType
    ) -> None:
        """Update occurrences of same values using a compressed sequence.

        Args:
            sequence: compressed sequence.
        """
        occurrences = deepcopy(self._occurrences)
        for patch in sequence:
            for value in patch:
                if value in occurrences.keys():
                    occurrences[value] += 1
                else:
                    occurrences[value] = 1
        self.occurrences = occurrences

    def add_instance(self, instance: Any) -> None:
        """Accumulates the occurrences of a given instance.

        Args:
            instance: instance to process.
        """

        sequence: OccurrencesSequenceType = self.compress(instance)
        self.update_occurrences_from_sequence(sequence)

    def add_multiple_instances(
        self, instances: List[Any], num_workers: int = 4
    ) -> None:
        """Accumulates the occurrences from multiple instances.

        Args:
            instances: instances to add.
            num_workers: number of workers to use for compression.
        """
        # parallelized compression of instances into sequences
        if num_workers >= 1:
            with mp.Pool(num_workers) as p:
                all_sequences = p.map(self.compress, instances)
        else:
            all_sequences = [self.compress(instance) for instance in instances]

        # (non parallel) accumulation of occurrences from all sequences
        for sequence in all_sequences:
            self.update_occurrences_from_sequence(sequence)

    def update_occurrences(
        self,
        occurrences: Optional[OrderedDict[OccurrenceType, int]] = None,
        overwrite: bool = True,
    ) -> None:
        """Update occurrences.

        This triggers look-up tables rebuilding.

        Args:
            occurrences: optional occurrences to use for the update. Defaults to None.
            overwrite: whether to overwrite or update existing occurrences. Defaults to True.
        """
        if occurrences is not None:
            if overwrite:
                updated_occurrences = deepcopy(occurrences)
            else:
                updated_occurrences = deepcopy(self._occurrences)
                for key in occurrences.keys():
                    updated_occurrences[key] = (
                        updated_occurrences.get(key, 0) + occurrences[key]
                    )
            self.occurrences = updated_occurrences

    @abstractmethod
    def compute_quality(self, instance: Any) -> float:
        """Measures encoding/decoding quality on an instance.

        Args:
            instance: instance to compute error

        Returns:
            quality measure.
        """


class UnicodeTextBWRLETransform(UnicodeDataTransform):
    """Unicode transform for textual data using the Burrows Wheeler Transform and Run Length Encoding."""

    @staticmethod
    def slice_substrings(input: str, substring_length: int) -> List[str]:
        """Helper function to slice a string into patches.

        Args:
            input: input string.
            substring_length: desired substring length, enough to benefit from RLE compression.

        Returns:
            A list of substrings in the right order to recompose the input.
        """

        substrings = [
            input[i : i + substring_length]
            for i in range(0, len(input), substring_length)
        ]

        return substrings

    @staticmethod
    def run_length_encoder(raw_string: str) -> str:
        """Run Length Encoding of a string.

        Args:
            raw_string: string to encode.

        Returns:
            RL-encoded string.
        """
        result = ""
        count = 1
        for i in range(len(raw_string) - 1):
            if raw_string[i] == raw_string[i + 1]:
                count += 1
            else:
                result += str(count) + raw_string[i] + "\u001f"
                count = 1
        result += str(count) + raw_string[i + 1]

        return result

    @staticmethod
    def run_length_decoder(encoded_strings: str) -> str:
        """Run Length Decoding of a string.

        Args:
            encoded_strings: RL-encoded string.

        Returns:
            decoded string.
        """

        decoded_string = ""
        for run in encoded_strings.split("\u001f"):
            try:
                count = int(run[:-1])
            except ValueError:
                count = 1
            character = run[-1]
            decoded_string += character * count

        return decoded_string

    @staticmethod
    def bwt_rle(input: str) -> str:
        """Burrows Wheeler Transform with Run Length Encoding.

        Args:
            input: string to encode.

        Returns:
            an encoded string with BW and RLE.
        """
        sorted_shifts = sorted(
            ["\u001d".join([input[i:], input[:i]]) for i in range(len(input) + 1)]
        )
        bwt = "".join([shift[-1] for shift in sorted_shifts])
        bwt_rle = UnicodeTextBWRLETransform.run_length_encoder(bwt)
        return bwt_rle

    @staticmethod
    def inverse_bwt_rle(encoded_input: str) -> str:
        """Inverse of the Burrows Wheeler Transform with Run Length Encoding.

        Args:
            encoded_input: encoded string.

        Returns:
            a decoded string.
        """
        bwt_input = UnicodeTextBWRLETransform.run_length_decoder(encoded_input)
        possibilities = [""] * len(bwt_input)

        for i in range(len(bwt_input)):
            new_possibility = [
                "".join([character, possibility])
                for character, possibility in zip(bwt_input, possibilities)
            ]
            possibilities = sorted(new_possibility)

        decoded_bwt = possibilities[0][1:]
        return decoded_bwt

    def compress(self, instance: str) -> OccurrencesSequenceType:
        """Compress an instance.

        Args:
            instance: instance to compress.

        Returns:
            the instance compressed as an occurrence sequence.
        """
        sequence: OccurrencesSequenceType = []
        for string_patch in UnicodeTextBWRLETransform.slice_substrings(
            instance, self.patch_size
        ):
            bwt_string_patch = UnicodeTextBWRLETransform.bwt_rle(string_patch)
            sequence.append(
                cast(List[OccurrenceType], bwt_string_patch.split("\u001f"))
            )
        return sequence

    def decompress(
        self, sequence: OccurrencesSequenceType, dimensions: Optional[List[int]] = None
    ) -> str:
        """Decompress a sequence in an instance.

        Args:
            sequence: sequence to decompress.
            dimensions: optional dimensions for decompression. Defaults to None.

        Returns:
            instance decompressed from an occurrence sequence.
        """
        reconstructed = []
        if dimensions is not None:
            logger.warning(f"Dimensions {dimensions} not used in {self.__class__}")
        for patch in sequence:
            reconstructed_patch = "\u001f".join(cast(List[str], patch))
            bwt_decoded_patch = UnicodeTextBWRLETransform.inverse_bwt_rle(
                reconstructed_patch
            )

            reconstructed.append(bwt_decoded_patch)
        return "".join(reconstructed)

    def compute_quality(self, instance: str) -> float:
        """Measures encoding/decoding quality on an instance.

        Args:
            instance: instance to compute error

        Returns:
            quality measure.
        """
        text = self.encode(instance.strip())
        reconstructed_signal = self.decode(text)

        quality = SequenceMatcher(None, instance.strip(), reconstructed_signal).ratio()
        return quality


class UnicodeJPEGTransform(UnicodeDataTransform):
    """Unicode transform using a JPEG-like compression method"""

    def __init__(
        self,
        configuration: UnicodeJPEGTransformParameters = UnicodeJPEGTransformParameters(),
    ):
        """Creates a unicode data transform.

        Args:
            patch_size: patch size for instance compression algorithm. Default to 16.
            offset_value: initial unicode character offset to use for pattern occurrences.
            compression_parameters: optional compression parameters.
        """
        super().__init__(configuration=configuration)
        self.max_coefficients = configuration.max_coefficients
        self.dct_precision = configuration.dct_precision

    def compress(self, instance: NDArray[Any]) -> OccurrencesSequenceType:
        """Compresses an instance into a list of lists of self.max_coefficients^2  DCT coefficients.

        Args:
            instance: input instance to compress.

        Returns:
            a list of lists of DCT coefficients of the compressed instance.
        """
        dimensions = list(instance.shape)

        if len(dimensions) == 3:
            channel_sequences = [
                self.compress(instance[:, :, channel])
                for channel in range(dimensions[2])
            ]
            sequence: List[Any] = []
            for patch in range(len(channel_sequences[0])):
                values: List[Any] = []
                for channel in range(dimensions[2]):
                    values = values + channel_sequences[channel][patch]
                sequence.append(values)
            return sequence
        else:
            width = dimensions[0]
            height = dimensions[1]

            updated_width = self.patch_size * np.ceil(width / self.patch_size)
            updated_height = self.patch_size * np.ceil(height / self.patch_size)
            instance = skimage.transform.resize(
                instance, (updated_width, updated_height)
            )

            dct_mask = np.zeros((self.patch_size, self.patch_size))
            dct_mask[0 : self.max_coefficients, 0 : self.max_coefficients] = 1
            blocks = view_as_blocks(instance, (self.patch_size, self.patch_size))
            sequence = []

            for col in range(blocks.shape[1]):
                for row in range(blocks.shape[0]):
                    block = blocks[row, col, :, :]
                    dct_block = dctn(block).round(self.dct_precision)
                    values_to_store = dct_block[
                        0 : self.max_coefficients, 0 : self.max_coefficients
                    ].astype(np.float16)

                    sequence.append(values_to_store.reshape((-1)).tolist())

            return sequence

    def decompress(
        self, sequence: OccurrencesSequenceType, dimensions: Optional[List[int]] = None
    ) -> NDArray[Any]:
        """Decompresses an instance from a list of lists of self.max_coefficients^2  DCT coefficients.

        Args:
            sequence: List of list of coefficients
            dimensions: original dimensions of the instance (H,W,C)

        Returns:
            reconstructed instance.
        """
        if dimensions is not None:
            width = dimensions[0]
            height = dimensions[1]
            if len(dimensions) == 3:
                channels = dimensions[2]
            else:
                channels = 1
        else:
            width = len(sequence)
            height = self.patch_size
            channels = 1

        dimensions = [width, height, channels]

        instance = np.zeros(dimensions)
        updated_width = self.patch_size * np.ceil(width / self.patch_size)
        updated_height = self.patch_size * np.ceil(height / self.patch_size)
        instance = skimage.transform.resize(
            instance, (updated_width, updated_height, channels)
        )
        dct_mask = np.zeros((self.patch_size, self.patch_size))

        for channel in range(channels):
            blocks = view_as_blocks(
                instance[:, :, channel], (self.patch_size, self.patch_size)
            )
            sequence_index = 0
            for col in range(blocks.shape[1]):
                for row in range(blocks.shape[0]):
                    values = sequence[sequence_index][
                        channel
                        * self.max_coefficients
                        * self.max_coefficients : (channel + 1)
                        * self.max_coefficients
                        * self.max_coefficients
                    ]
                    arranged_values = np.asarray(values).reshape(
                        (
                            self.max_coefficients,
                            self.max_coefficients,
                        )
                    )
                    dct_mask[
                        0 : self.max_coefficients, 0 : self.max_coefficients
                    ] = arranged_values
                    idct_block = idctn(dct_mask)
                    instance[
                        row * self.patch_size : (row + 1) * self.patch_size,
                        col * self.patch_size : (col + 1) * self.patch_size,
                        channel,
                    ] = idct_block
                    sequence_index += 1

        instance = skimage.transform.resize(instance, (width, height, channels))
        instance = instance - np.min(instance)
        instance = instance / np.max(instance)

        return np.squeeze(instance)

    def compute_quality(self, instance: NDArray[np.float64]) -> float:
        """Measures encoding/decoding quality on an instance.

        Args:
            instance: instance to compute error

        Returns:
            quality measure.
        """
        signal = np.asarray(instance)
        text = self.encode(signal)
        reconstructed_signal = self.decode(text, list(signal.shape))

        mse = np.mean(
            (
                signal / np.max(signal)
                - reconstructed_signal / np.max(reconstructed_signal)
            )
            ** 2
        )
        if mse == 0:  # MSE is zero means no noise is present in the signal .
            # Therefore PSNR have no importance.
            return 100.0
        psnr = float(20 * np.log10(1 / np.sqrt(mse)))
        return psnr

class UnicodeSeriesCompansionTransform(UnicodeDataTransform):
    """Unicode transform for sequences of floats using companding."""

    def __init__(
        self,
        configuration: UnicodeSeriesCompansionTransformParameters = UnicodeSeriesCompansionTransformParameters(),
    ):
        """Creates a unicode data transform.

        Args:
            patch_size: patch size for instance compression algorithm. Default to 16.
            offset_value: initial unicode character offset to use for pattern occurrences.
            compression_parameters: optional compression parameters.
        """
        super().__init__(configuration=configuration)
        self.companding_max = configuration.companding_max
        self.mu = configuration.mu
        self.float_precision = configuration.float_precision

    def float_to_int(self, float_number: float) -> int:
        """Convert float to int with a given precision.

        Args:
            float_number: float to convert.

        Returns:
            converted number as integer.
        """
        scaling = (2**self.float_precision + 1) - 1
        integer_number = int(scaling * float_number)
        return integer_number

    def int_to_float(self, integer_number: int) -> float:
        """Convert int to float with a given precision.

        Args:
            integer_number: integer to convert.

        Returns:
            converted number as float.
        """
        scaling: float = (2**self.float_precision + 1) - 1
        float_number: float = integer_number / scaling
        return float_number

    def compand(self, data: NDArray[np.float64]) -> NDArray[np.int64]:
        """Apply companding.

        Args:
            data: data to compand.

        Returns:
            companded data.
        """
        data = np.array(data) / self.companding_max
        data = np.clip(data, -1.0, 1.0)

        companded_data = np.sign(data) * np.log(1 + self.mu * np.abs(data)) / np.log(1 + self.mu)
        companded_list = [self.float_to_int(x) for x in companded_data.reshape(-1).tolist()]
        # NOTE: companded_array = companded_data.reshape(-1).apply(self.float_to_int)
        return cast(NDArray[np.int64], np.asarray(companded_list).squeeze())

    def expand(self, data: NDArray[np.int64]) -> NDArray[np.float64]:
        """Expand signal dynamic range using Mu law.

        Args:
            data: data to expand.

        Returns:
            expanded data.
        """
        data = np.asarray([self.int_to_float(int(x)) for x in data.reshape(-1).tolist()])
        expanded_data: NDArray[Any]
        expanded_data = np.sign(data) * (np.power((1 + self.mu), np.abs(data)) - 1) / self.mu
        expanded_data = np.clip(expanded_data, -1, 1) * self.companding_max
        return cast(NDArray[np.float64], expanded_data.squeeze())

    def compress(self, instance: NDArray[Any]) -> OccurrencesSequenceType:
        """Compresses a float series into a sequence by companding.

        Args:
            instance: input sequence to compress..

        Returns:
            a list of lists of coefficients of the compressed sequence.
        """

        companded = self.compand(instance)
        new_dimension = int(self.patch_size * np.ceil(companded.size / self.patch_size))
        companded = np.pad(companded, (0, new_dimension - len(instance)), mode="edge")
        reshaped_companded = companded.reshape((-1, self.patch_size))
        sequence = []
        for patch in reshaped_companded:
            sequence.append(list(patch))
        return cast(OccurrencesSequenceType, sequence)

    def decompress(
        self, sequence: OccurrencesSequenceType, dimensions: Optional[List[int]] = None
    ) -> NDArray[Any]:
        """Decompresses an instance from a sequence by expanding.

        Args:
            sequence: list of list of coefficients.
            dimensions: unused.

        Returns:
            reconstructed instance.
        """
        if dimensions is not None:
            logger.debug(f"Dimensions {dimensions} not used")
        reconstructed = np.zeros((len(sequence), self.patch_size), dtype=np.int64)
        for sequence_id, patch in enumerate(sequence):
            reconstructed[sequence_id, :] = patch
        return np.squeeze(self.expand(reconstructed))

    def compute_quality(self, instance: NDArray[np.float64]) -> float:
        """Measures encoding/decoding quality on an instance.

        Args:
            instance: instance to compute error

        Returns:
            quality measure.
        """
        compressed_instance = self.compress(instance)
        decompressed_instance = self.decompress(compressed_instance)[: len(instance)]
        encoded_instance = self.encode(instance)
        decoded_instance = self.decode(encoded_instance)[: len(instance)]
        compression_error = np.mean(np.power(instance - decompressed_instance, 2))
        encoding_error = np.mean(np.power(instance - decoded_instance, 2))
        if (encoding_error / compression_error + 1e-12) > 1.05:
            logger.warning("Encoding error higher than compression error")
        mse = encoding_error
        if mse == 0:  # MSE is zero means no noise is present in the instance .
            # Therefore PSNR have no importance.
            return 100
        psnr = float(20 * np.log10(self.companding_max / np.sqrt(mse)))
        return psnr
    

UNICODE_TRANSFORMS: Dict[str, Type[UnicodeDataTransform]] = {
    "text_bwrle": UnicodeTextBWRLETransform,
    "wikitext": UnicodeTextBWRLETransform,
    "jpeg": UnicodeJPEGTransform,
    "pass": UnicodeJPEGTransform,
    "cifar10": UnicodeJPEGTransform,
    "compansion": UnicodeSeriesCompansionTransform,
    UnicodeTextBWRLETransform.__name__: UnicodeTextBWRLETransform,
    UnicodeJPEGTransform.__name__: UnicodeJPEGTransform,
    UnicodeSeriesCompansionTransform.__name__: UnicodeSeriesCompansionTransform,
}


UNICODE_CONFIGURATIONS: Dict[str, Type[UnicodeTransformParameters]] = {
    "text_bwrle": UnicodeTextBWRLETransformParameters,
    "wikitext": UnicodeTextBWRLETransformParameters,
    "jpeg": UnicodeJPEGTransformParameters,
    "pass": UnicodeJPEGTransformParameters,
    "cifar10": UnicodeJPEGTransformParameters,
    "compansion": UnicodeSeriesCompansionTransformParameters,
    UnicodeTextBWRLETransform.__name__: UnicodeTextBWRLETransformParameters,
    UnicodeJPEGTransform.__name__: UnicodeJPEGTransformParameters,
    UnicodeSeriesCompansionTransform.__name__: UnicodeSeriesCompansionTransformParameters,
}
