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
"""Helpers for testing"""

import json
import os
import shutil
import tarfile
from itertools import product, starmap

import numpy as np
from PIL import Image
import ray

TEST_PATH = "./testing_tmp"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TQDM_ENV_VAR = "DISBALE_TQDM"


class MockTQDM:
    unit = ""

    def update(self, n: int = 1):
        pass

    def clear(self):
        pass

    def display(self, message):
        pass

    def set_description(self, desc):
        pass

    def reset(total):
        pass


class MockWCS:
    """Mock WCS object for testing"""

    def __init__(self, include_cd: bool):
        if include_cd:
            self.cd = np.array([[1, 0], [0, 1]])
        else:
            self.crpix = np.array([1, 1])
            self.crval = np.array([1, 1])

    def all_pix2world(self, *args, **kwargs):
        return np.array([[1, 1], [1, 1], [1, 1]]).astype(np.float64)

    @property
    def wcs(self):
        return self


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

        def extract(f):
            with tarfile.open(with_test_path(f)) as f:
                f.extractall(TEST_PATH)

        any(map(extract, compressed_files))


def tear_down(include_ray=False):
    """Tears down testing structure"""

    if os.path.exists(TEST_PATH):
        shutil.rmtree(TEST_PATH)

    if include_ray:
        ray.shutdown()


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


def compare_file_directories(dir1, dir2) -> bool:
    is_file = lambda x: x.is_file()
    is_dir = lambda x: x.is_dir()
    get_name = lambda x: x.name
    get_path = lambda x: x.path

    def get_file_extension(fname):
        return os.path.splitext(fname)[1]

    def compare_file_contents(file1, file2) -> bool:
        f_ext = get_file_extension(file1)
        if f_ext in [".png", ".tiff", ".ico"]:
            arr1 = np.array(Image.open(file1))
            arr2 = np.array(Image.open(file2))
            same = np.isclose(
                arr1,
                arr2,
                rtol=1e-05,
                atol=5,  # these are integer images so 15 is a reasonable tolerance
                equal_nan=True,
            )

            if not same.all():
                ys, xs, cs = np.where(~same)
                print(f"Found {len(ys)} differences in {file1}")
                print(f"First difference at {ys[0]}, {xs[0]}, {cs[0]}")
                print(
                    f"First difference is {arr1[ys[0], xs[0], cs[0]]} vs {arr2[ys[0], xs[0], cs[0]]}"
                )
                return False

            return same.all()
        else:
            mode = "r" + "b" * int(f_ext in [".cbor", ".pbf"])
            with open(file1, mode) as f1, open(file2, mode) as f2:
                try:
                    return f1.readlines() == f2.readlines()
                except:
                    print(file1, file2, mode)

    def compare_subdirs(sub_dir1, sub_dir2) -> bool:
        dir1_entries = list(os.scandir(sub_dir1))
        dir2_entries = list(os.scandir(sub_dir2))

        dir1_files = list(sorted(filter(is_file, dir1_entries), key=lambda x: x.name))
        dir2_files = list(sorted(filter(is_file, dir2_entries), key=lambda x: x.name))

        count_and_names_same = list(map(get_name, dir1_files)) == list(
            map(get_name, dir2_files)
        )

        if count_and_names_same:
            # compare file contents
            file_pairs = list(zip(map(get_path, dir1_files), map(get_path, dir2_files)))
            files_comps = list(
                starmap(
                    compare_file_contents,
                    zip(
                        map(get_path, dir1_files),
                        map(get_path, dir2_files),
                    ),
                )
            )

            files_same = all(files_comps)

            if files_same:
                dir1_subdirs = list(
                    sorted(filter(is_dir, dir1_entries), key=lambda x: x.name)
                )
                dir2_subdirs = list(
                    sorted(filter(is_dir, dir2_entries), key=lambda x: x.name)
                )

                count_and_names_same = list(map(get_name, dir1_subdirs)) == list(
                    map(get_name, dir2_subdirs)
                )

                if count_and_names_same:
                    # compare sub dirs
                    subdir_pairs = list(
                        zip(
                            map(get_path, dir1_subdirs),
                            map(get_path, dir2_subdirs),
                        )
                    )
                    subdir_comp = list(starmap(compare_subdirs, subdir_pairs))
                    subdirs_same = all(subdir_comp)

                    return subdirs_same
                else:
                    list(
                        map(
                            lambda x: print(x[1], "don't match"),
                            filter(lambda x: not x[0], zip(subdir_comp, subdir_pairs)),
                        )
                    )
                    return False
            else:
                list(
                    map(
                        lambda x: print(x[1], "don't match"),
                        filter(lambda x: not x[0], zip(files_comps, file_pairs)),
                    )
                )
                return False
        else:
            missing_files = set(
                list(map(lambda f: f.name, dir1_files))
            ).symmetric_difference(set(list(map(lambda f: f.name, dir2_files))))
            print(missing_files)
            return False

    return compare_subdirs(dir1, dir2)


def get_version():
    here = os.path.dirname(os.path.realpath(__file__))
    version_lcocation = os.path.join(here, "../__version__.py")

    with open(version_lcocation, "r") as f:
        return f.readline().strip().replace('"', "")
