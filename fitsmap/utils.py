# MIT License
# Copyright 2021 Ryan Hausen and contributers

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import os
import string
from functools import reduce
from itertools import chain, filterfalse
from typing import Iterable, List, Tuple

from astropy.io import fits
from tqdm import tqdm
from PIL import Image

import fitsmap


def digit_to_string(digit: int) -> str:
    """Converts an integer into its word representation"""

    if digit == 0:
        return "zero"
    elif digit == 1:
        return "one"
    elif digit == 2:
        return "two"
    elif digit == 3:
        return "three"
    elif digit == 4:
        return "four"
    elif digit == 5:
        return "five"
    elif digit == 6:
        return "six"
    elif digit == 7:
        return "seven"
    elif digit == 8:
        return "eight"
    elif digit == 9:
        return "nine"
    else:
        raise ValueError("Only digits 0-9 are supported")


def make_fname_js_safe(fname: str) -> str:
    """Converts a string filename to a javascript safe identifier."""

    if fname[0] in string.digits:
        adj_for_digit = digit_to_string(int(fname[0])) + fname[1:]
    else:
        adj_for_digit = fname

    return adj_for_digit.replace(".", "_dot_").replace("-", "_")


def get_fits_image_size(fits_file: str) -> Tuple[int, int]:
    """Returns image size (x, y)

    Args:
        fits_file (str): fits file path

    Returns:
        Tuple[int, int]: returns the x and y dims of the input file
    """
    hdr = fits.getheader(fits_file)
    return hdr["NAXIS1"], hdr["NAXIS2"]


def get_standard_image_size(image_file: str) -> Tuple[int, int]:
    """Returns image size (x, y)

    Args:
        image_file (str): image file path

    Returns:
        Tuple[int, int]: returns the x and y dims of the input file
    """
    with Image.open(image_file) as f:
        size = f.size

    return size


def peek_image_info(img_file_names: List[str]) -> Tuple[int, int]:
    """Gets image size values given passed image file names

    Args:
        img_file_names (List[str]): Input image files that are being tiled

    Returns:
        Tuple[int, int]: The `max x`, and `max y`
    """

    fits_sizes = list(
        map(get_fits_image_size, filter(lambda f: f.endswith("fits"), img_file_names),)
    )

    standard_sizes = list(
        map(
            get_standard_image_size,
            filterfalse(lambda f: f.endswith("fits"), img_file_names),
        )
    )

    max_x, max_y = reduce(
        lambda x, y: (max(x[0], y[0]), max(x[1], y[1])),
        chain.from_iterable([fits_sizes, standard_sizes]),
        (0, 0),
    )

    return max_x, max_y


def get_version():
    with open(os.path.join(fitsmap.__path__[0], "__version__.py"), "r") as f:
        return f.readline().strip().replace('"', "")
