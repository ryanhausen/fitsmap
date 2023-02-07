# MIT License
# Copyright 2023 Ryan Hausen and contributers

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

from typing import Any, Tuple, Union

import numpy as np


class PaddedArray:
    def __init__(self, array: np.ndarray, pad: Tuple[int, int], tile_size=(256, 256)):
        self.array = array
        self.pad = pad

        shape = [array.shape[0] + pad[0], array.shape[1] + pad[1]]
        empty_tile_shape = list(tile_size)
        if len(array.shape) == 3:
            shape += [array.shape[2]]
            empty_tile_shape += [array.shape[2]]

        self.empty_tile = np.full(empty_tile_shape, np.nan, dtype=np.float32)
        self.shape = tuple(shape)

    def __get_internal_array(self, ys: slice, xs: slice) -> np.ndarray:
        return self.array[ys, xs]

    # def __get_padding(self, ys: slice, xs: slice) -> np.ndarray:
    #     shape = [ys.stop - ys.start, xs.stop - xs.start]
    #     if len(self.shape) == 3:
    #         shape += [self.shape[2]]
    #     return np.full(shape, np.nan, dtype=np.float32)

    def __get_mixed(self, ys: slice, xs: slice) -> np.ndarray:
        start_y, stop_y = ys.start, ys.stop
        start_x, stop_x = xs.start, xs.stop

        pad_y = max(0, stop_y - self.array.shape[0])
        pad_x = max(0, stop_x - self.array.shape[1])

        padding = [[0, pad_y], [0, pad_x]]
        if len(self.array.shape) == 3:
            padding += [[0, 0]]

        slice_ys = slice(start_y, min(self.array.shape[0], stop_y))
        slice_xs = slice(start_x, min(self.array.shape[1], stop_x))

        return np.pad(
            self.array[slice_ys, slice_xs],
            padding,
            mode="constant",
            constant_values=np.nan,
        )

    def __getitem__(self, axis_slices: Tuple[slice, slice]) -> np.ndarray:
        if type(axis_slices) is not tuple:
            axis_slices = [axis_slices, slice(None)]

        ys, xs = axis_slices
        start_y, stop_y = ys.start or 0, ys.stop or self.shape[0]
        start_x, stop_x = xs.start or 0, xs.stop or self.shape[1]

        slice_ys = slice(start_y, stop_y)
        slice_xs = slice(start_x, stop_x)

        if stop_y < self.array.shape[0] and stop_x < self.array.shape[1]:
            return self.__get_internal_array(slice_ys, slice_xs)
        elif start_y > self.array.shape[0] or start_x > self.array.shape[1]:
            # this is a hack because in our use case we know we always
            # want the same sized thing. In other use cases the commented out
            # function should be called
            return self.empty_tile
            # return self.__get_padding(slice_ys, slice_xs)
        else:
            return self.__get_mixed(slice_ys, slice_xs)

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        return PaddedArray, (self.array, self.pad)
