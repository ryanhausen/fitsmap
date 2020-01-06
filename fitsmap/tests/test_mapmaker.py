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
"""Tests mapmaker.py"""

import os

import numpy as np
import pytest
from astropy.io import fits
from PIL import Image
from skimage.data import camera

import fitsmap.mapmaker as mm
import fitsmap.tests.helpers as helpers


@pytest.mark.unit
def test_build_path():
    """Test the mapmaker.build_path function"""
    z, y, x = 1, 2, 3
    out_dir = helpers.TEST_PATH
    img_name = mm.build_path(z, y, x, out_dir)

    expected_img_name = os.path.join(out_dir, str(z), str(y), f"{x}.png")

    expected_file_name_matches = expected_img_name == img_name

    assert expected_file_name_matches


@pytest.mark.unit
def test_slice_idx_generator_z0():
    """Test mapmaker.slice_idx_generator at zoom level 0.

    The given shape (4305, 9791) breaks iterative schemes that don't properly
    seperate tiles. Was a bug.
    """
    shape = (4305, 9791)
    zoom = 0
    given = mm.slice_idx_generator(shape, zoom)
    expected = helpers.get_slice_idx_generator_solution(zoom)

    comparable_given = set(map(helpers.covert_idx_to_hashable_tuple, given))
    comparable_expected = set(map(helpers.covert_idx_to_hashable_tuple, expected))

    assert comparable_given == comparable_expected


@pytest.mark.unit
def test_slice_idx_generator_z1():
    """Test mapmaker.slice_idx_generator at zoom level 1.

    The given shape (4305, 9791) breaks iterative schemes that don't properly
    seperate tiles. Was a bug.
    """
    shape = (4305, 9791)
    zoom = 1
    given = mm.slice_idx_generator(shape, zoom)
    expected = helpers.get_slice_idx_generator_solution(zoom)

    comparable_given = set(map(helpers.covert_idx_to_hashable_tuple, given))
    comparable_expected = set(map(helpers.covert_idx_to_hashable_tuple, expected))

    assert comparable_given == comparable_expected


@pytest.mark.unit
def test_slice_idx_generator_z2():
    """Test mapmaker.slice_idx_generator at zoom level 2.

    The given shape (4305, 9791) breaks iterative schemes that don't properly
    seperate tiles. Was a bug.
    """
    shape = (4305, 9791)
    zoom = 2
    given = mm.slice_idx_generator(shape, zoom)
    expected = helpers.get_slice_idx_generator_solution(zoom)

    comparable_given = set(map(helpers.covert_idx_to_hashable_tuple, given))
    comparable_expected = set(map(helpers.covert_idx_to_hashable_tuple, expected))

    assert comparable_given == comparable_expected


@pytest.mark.unit
def test_slice_idx_generator_z3():
    """Test mapmaker.slice_idx_generator at zoom level 3.

    The given shape (4305, 9791) breaks iterative schemes that don't properly
    seperate tiles. Was a bug.
    """
    shape = (4305, 9791)
    zoom = 3
    given = mm.slice_idx_generator(shape, zoom)
    expected = helpers.get_slice_idx_generator_solution(zoom)

    comparable_given = set(map(helpers.covert_idx_to_hashable_tuple, given))
    comparable_expected = set(map(helpers.covert_idx_to_hashable_tuple, expected))

    assert comparable_given == comparable_expected


@pytest.mark.unit
def test_slice_idx_generator_z4():
    """Test mapmaker.slice_idx_generator at zoom level 4.

    The given shape (4305, 9791) breaks iterative schemes that don't properly
    seperate tiles. Was a bug.
    """
    shape = (4305, 9791)
    zoom = 4
    given = mm.slice_idx_generator(shape, zoom)
    expected = helpers.get_slice_idx_generator_solution(zoom)

    comparable_given = set(map(helpers.covert_idx_to_hashable_tuple, given))
    comparable_expected = set(map(helpers.covert_idx_to_hashable_tuple, expected))

    assert comparable_given == comparable_expected


@pytest.mark.unit
def test_slice_idx_generator_z5():
    """Test mapmaker.slice_idx_generator at zoom level 5.

    The given shape (4305, 9791) breaks iterative schemes that don't properly
    seperate tiles. Was a bug.
    """
    shape = (4305, 9791)
    zoom = 5
    given = mm.slice_idx_generator(shape, zoom)
    expected = helpers.get_slice_idx_generator_solution(zoom)

    comparable_given = set(map(helpers.covert_idx_to_hashable_tuple, given))
    comparable_expected = set(map(helpers.covert_idx_to_hashable_tuple, expected))

    assert comparable_given == comparable_expected


@pytest.mark.unit
def test_balance_array_2d():
    """Test mapmaker.balance_array"""

    in_shape = (10, 20)
    expected_shape = (20, 20)
    expected_num_nans = 200

    test_array = np.zeros(in_shape)

    out_array = mm.balance_array(test_array)

    assert out_array.shape == expected_shape
    assert np.isnan(out_array).sum() == expected_num_nans


@pytest.mark.unit
def test_balance_array_3d():
    """Test mapmaker.balance_array"""

    in_shape = (10, 20, 3)
    expected_shape = (20, 20, 3)
    expected_num_nans = 600

    test_array = np.zeros(in_shape)

    out_array = mm.balance_array(test_array)

    assert out_array.shape == expected_shape
    assert np.isnan(out_array).sum() == expected_num_nans


@pytest.mark.unit
def test_get_array_fits():
    """Test mapmaker.get_array"""

    helpers.setup()

    # make test array
    expected_array = np.zeros((3, 3))
    out_path = os.path.join(helpers.TEST_PATH, "test.fits")
    fits.PrimaryHDU(data=expected_array).writeto(out_path)

    # get test array
    actual_array = mm.get_array(out_path)

    helpers.tear_down()

    assert np.array_equal(expected_array, actual_array)


@pytest.mark.unit
def test_get_array_png():
    """Test mapmaker.get_array"""

    helpers.setup()

    # make test array
    expected_array = camera()
    out_path = os.path.join(helpers.TEST_PATH, "test.png")
    Image.fromarray(expected_array).save(out_path)

    # get test array
    actual_array = mm.get_array(out_path)

    helpers.tear_down()

    assert np.array_equal(expected_array, np.flipud(actual_array))


@pytest.mark.unit
def test_filter_on_extension_without_predicate():
    """Test mapmaker.filter_on_extension without a predicate argument"""

    test_files = ["file_one.fits", "file_two.fits", "file_three.exclude"]

    extensions = ["fits"]
    expected_list = test_files[:-1]

    actual_list = mm.filter_on_extension(test_files, extensions)

    assert expected_list == actual_list


@pytest.mark.unit
def test_filter_on_extension_with_predicate():
    """Test mapmaker.filter_on_extension with a predicate argument"""

    test_files = ["file_one.fits", "file_two.fits", "file_three.exclude"]

    extensions = ["fits"]
    expected_list = test_files[:1]
    predicate = lambda f: f == test_files[1]

    actual_list = mm.filter_on_extension(test_files, extensions, predicate)

    assert expected_list == actual_list


@pytest.mark.unit
def test_make_dirs():
    """Test mapmaker.make_dirs"""

    helpers.setup()

    out_dir = helpers.TEST_PATH

    expected_dirs = set(
        list(
            map(
                lambda f: os.path.join(out_dir, f),
                ["0/0", "1/0", "1/1", "2/0", "2/1", "2/2", "2/3"],
            )
        )
    )

    mm.make_dirs(out_dir, 0, 2)

    dirs_exists = all(map(os.path.exists, expected_dirs))

    helpers.tear_down()

    assert dirs_exists


@pytest.mark.unit
def test_get_zoom_range():
    """Test mapmaker.get_zoom_range"""

    in_shape = [10000, 10000]
    tile_size = [256, 256]

    expected_min = 0
    expected_max = 5

    actual_min, acutal_max = mm.get_zoom_range(in_shape, tile_size)

    assert expected_min == actual_min
    assert expected_max == acutal_max

@pytest.mark.unit
def test_get_total_tiles():
    """Test mapmaker.get_total_tiles"""

    min_zoom, max_zoom = 0, 2
    expected_number = 21

    actual_number = mm.get_total_tiles(min_zoom, max_zoom)
    assert expected_number == actual_number