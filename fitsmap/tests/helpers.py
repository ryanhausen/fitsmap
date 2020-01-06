# MIT License
# Copyright 2019 Ryan Hausen

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
"""Helpers for testing"""

import os
import shutil
from functools import reduce
from itertools import chain, product, repeat

TEST_PATH = "./testing_tmp"


def setup():
    """Builds testing structure"""

    if not os.path.exists(TEST_PATH):
        os.mkdir(TEST_PATH)


def tear_down():
    """Tears down testing structure"""

    if os.path.exists(TEST_PATH):
        shutil.rmtree(TEST_PATH)


def __stable_idx_answer(shape, zoom):
    num_splits = int((4 ** zoom) ** 0.5)

    def split(vals):
        x0, x2 = vals
        x1 = x0 + ((x2 - x0) // 2)
        return [(x0, x1), (x1, x2)]

    split_collection = lambda collection: map(split, collection)
    split_reduce = lambda x, y: split_collection(chain.from_iterable(x))

    rows_split = list(reduce(split_reduce, repeat(None, zoom), [[(0, shape[0])]]))
    columns_split = list(reduce(split_reduce, repeat(None, zoom), [[(0, shape[1])]]))

    rows = zip(range(num_splits - 1, -1, -1), chain.from_iterable(rows_split))
    cols = enumerate(chain.from_iterable(columns_split))

    rows_cols = product(rows, cols)

    def transform_iteration(row_col):
        ((y, (start_y, end_y)), (x, (start_x, end_x))) = row_col
        return (zoom, y, x, slice(start_y, end_y), slice(start_x, end_x))

    return map(transform_iteration, rows_cols)


def covert_idx_to_hashable_tuple(idx):
    """Converts idxs to hashable type for set, slice is not hashable"""
    return (idx[0], idx[1], idx[2], str(idx[3]), str(idx[4]))


def get_slice_idx_generator_solution(zoom: int):
    """Gets proper idxs using a method that tests correctly.

    The data returned by this can be big at high zoom levels.
    TODO: Find particular cases to test for.
    """
    return list(__stable_idx_answer((4305, 9791), zoom))
