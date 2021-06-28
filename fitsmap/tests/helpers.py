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

import filecmp
import json
import os
import shutil
import tarfile
from functools import reduce
from itertools import chain, product, repeat

import numpy as np

TEST_PATH = "./testing_tmp"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TQDM_ENV_VAR = "DISBALE_TQDM"


class MockTQDM:
    def update(self):
        pass


def setup(with_data=False):
    """Builds testing structure"""

    if not os.path.exists(TEST_PATH):
        os.mkdir(TEST_PATH)

    if with_data:
        with_data_path = lambda f: os.path.join(DATA_DIR, f)
        with_test_path = lambda f: os.path.join(TEST_PATH, f)

        copy_file = lambda f: shutil.copy(with_data_path(f), with_test_path(f))
        list(map(copy_file, os.listdir(DATA_DIR)))

        compressed_files = list(
            filter(lambda f: f.endswith("tar.xz"), os.listdir(TEST_PATH))
        )

        print()

        def extract(f):
            with tarfile.open(with_test_path(f)) as f:
                f.extractall(TEST_PATH)

        any(map(extract, compressed_files))


def tear_down():
    """Tears down testing structure"""

    if os.path.exists(TEST_PATH):
        shutil.rmtree(TEST_PATH)


def disbale_tqdm():
    os.environ[TQDM_ENV_VAR] = "True"


def enable_tqdm():
    os.environ[TQDM_ENV_VAR] = "False"


def cat_to_json(fname):
    with open(fname, "r") as f:
        lines = f.readlines()

    data = json.loads("[" + "".join([l.strip() for l in lines[1:-1]]) + "]")

    return data, lines[0]


def __stable_idx_answer(shape, zoom, tile_size=256):
    dim0_tile_fraction = shape[0] / tile_size
    dim1_tile_fraction = shape[1] / tile_size

    print(shape, dim0_tile_fraction, dim1_tile_fraction, tile_size, zoom)

    if dim0_tile_fraction < 1 or dim1_tile_fraction < 1:
        raise StopIteration()

    num_tiles_dim0 = int(np.ceil(dim0_tile_fraction))
    num_tiles_dim1 = int(np.ceil(dim1_tile_fraction))

    tile_idxs_dim0 = [i * tile_size for i in range(num_tiles_dim0 + 1)]
    tile_idxs_dim1 = [i * tile_size for i in range(num_tiles_dim1 + 1)]

    pair_runner = lambda coll: [slice(c0, c1) for c0, c1 in zip(coll[:-1], coll[1:])]

    row_slices = pair_runner(tile_idxs_dim0)
    col_slices = pair_runner(tile_idxs_dim1)

    rows = zip(range(num_tiles_dim0 - 1, -1, -1), row_slices)
    cols = enumerate(col_slices)

    rows_cols = product(rows, cols)

    def transform_iteration(row_col):
        ((y, slice_y), (x, slice_x)) = row_col
        return (zoom, y, x, slice_y, slice_x)

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


def compare_file_directories(dir1, dir2):
    def recursive_diff(dir_compare):
        if dir_compare.subdirs:
            print(dir_compare.subdirs)
            sub_match = all(map(recursive_diff, dir_compare.subdirs.values()))
        else:
            sub_match = True

        return sub_match and len(dir_compare.diff_files) == 0

    return recursive_diff(filecmp.dircmp(dir1, dir2))


def get_version():
    here = os.path.dirname(os.path.realpath(__file__))
    version_lcocation = os.path.join(here, "../__version__.py")

    with open(version_lcocation, "r") as f:
        return f.readline().strip().replace('"', "")
