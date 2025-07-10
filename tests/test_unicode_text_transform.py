"""Tests for ULT text transforms."""

from ult.configuration import UnicodeTextBWRLETransformParameters
from ult.transforms import UnicodeTextBWRLETransform

# Sample texts from Lewis Carrol's Alice in Wonderland.
TRAINING_TEXTS = [
    "CHAPTER I.",
    "Down the Rabbit-Hole",
    "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, “and what is the use of a book,” thought Alice “without pictures or conversations?”",
    "So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.",
    "There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, “Oh dear! Oh dear! I shall be late!” (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.",
    "In another moment down went Alice after it, never once considering how in the world she was to get out again.",
]
TEST_TEXT = "The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well."


def test_initialize() -> None:
    """Function to test initialize."""

    unicoder = UnicodeTextBWRLETransform()
    assert unicoder is not None


def test_unicode_text_bwtrle_transform() -> None:
    """Tests if a set of strings is properly encoded/decoded"""
    parameters = UnicodeTextBWRLETransformParameters()
    unicoder = UnicodeTextBWRLETransform(configuration=parameters)
    unicoder.add_multiple_instances(TRAINING_TEXTS)

    for signal in TRAINING_TEXTS:
        text = unicoder.encode(signal)
        reconstructed_signal = unicoder.decode(text)
        assert signal == reconstructed_signal


def test_bwrle_transform_quality() -> None:
    """
    Verify if the Quality can be computed
    """

    parameters = UnicodeTextBWRLETransformParameters()
    unicoder = UnicodeTextBWRLETransform(configuration=parameters)
    unicoder.add_multiple_instances(TRAINING_TEXTS)
    print(unicoder.decode(unicoder.encode(TEST_TEXT)))
    assert unicoder.compute_quality(TEST_TEXT) > 0
