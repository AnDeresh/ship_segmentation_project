import numpy as np
from typing import Tuple

def rle_decode(mask_rle: str, shape: Tuple[int, int] = (768, 768)) -> np.ndarray:
    """
    Decode run-length encoded (RLE) string to a binary mask.

    Args:
        mask_rle (str): Run-length as string formatted (start length)
        shape (Tuple[int, int], optional): Shape of the array to return. Defaults to (768, 768).

    Returns:
        np.ndarray: Decoded binary mask where 1 indicates mask and 0 indicates background.
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T