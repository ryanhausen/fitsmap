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
import string
from typing import Iterable

from tqdm import tqdm


class ShardedProcBarIter:
    """Maintains a single tqdm progress bar over multiple catalog shards.

    This is a helper class that keeps a single tqdm progress bar object for
    multiple shards of the same catalog.

    Attributes:
        iter (Iterable): the iterable that will be sharded
        proc_bar (tqdm): the tqdm object
    """

    def __init__(self, iter: Iterable, proc_bar: tqdm):
        self.iter = iter
        self.proc_bar = proc_bar

    def __iter__(self):
        return self

    def __next__(self):
        self.proc_bar.update()
        return next(self.iter)


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