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
"""Tests convert.py"""

import os
import warnings

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image
from multiprocessing import JoinableQueue
from skimage.data import camera

import fitsmap.convert as convert
import fitsmap.tests.helpers as helpers


@pytest.mark.unit
def test_build_path():
    """Test the convert.build_path function"""
    z, y, x = 1, 2, 3
    out_dir = helpers.TEST_PATH
    img_name = convert.build_path(z, y, x, out_dir)

    expected_img_name = os.path.join(out_dir, str(z), str(y), f"{x}.png")

    expected_file_name_matches = expected_img_name == img_name

    assert expected_file_name_matches


@pytest.mark.unit
def test_make_fname_js_safe_digit():
    """Test the convert.make_fname_js_safe functions."""

    unsafe = "123"
    expected = "one23"

    assert expected == convert.make_fname_js_safe(unsafe)


@pytest.mark.unit
def test_make_fname_js_safe_dot_dash():
    """Test the convert.make_fname_js_safe functions."""

    unsafe = "a.b-c"
    expected = "a_dot_b_c"

    assert expected == convert.make_fname_js_safe(unsafe)


@pytest.mark.unit
def test_make_fname_js_safe_no_change():
    """Test the convert.make_fname_js_safe functions."""

    safe = "abc"
    expected = "abc"

    assert expected == convert.make_fname_js_safe(safe)


@pytest.mark.unit
def test_digit_to_string():
    """Test the convert.digit_to_string function"""
    digits = range(10)
    strings = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    ]

    for expected, actual in zip(strings, map(convert.digit_to_string, digits)):
        assert expected == actual


@pytest.mark.unit
def test_slice_idx_generator_z0():
    """Test convert.slice_idx_generator at zoom level 0.

    The given shape (4305, 9791) breaks iterative schemes that don't properly
    seperate tiles. Was a bug.
    """
    shape = (4305, 9791)
    zoom = 0
    tile_size = 256
    given = convert.slice_idx_generator(shape, zoom, tile_size)
    expected = helpers.get_slice_idx_generator_solution(zoom)

    comparable_given = set(map(helpers.covert_idx_to_hashable_tuple, given))
    comparable_expected = set(map(helpers.covert_idx_to_hashable_tuple, expected))

    assert comparable_given == comparable_expected


@pytest.mark.unit
def test_slice_idx_generator_z1():
    """Test convert.slice_idx_generator at zoom level 1.

    The given shape (4305, 9791) breaks iterative schemes that don't properly
    seperate tiles. Was a bug.
    """
    shape = (4305, 9791)
    zoom = 1
    tile_size = 256
    given = convert.slice_idx_generator(shape, zoom, tile_size)
    expected = helpers.get_slice_idx_generator_solution(zoom)

    comparable_given = set(map(helpers.covert_idx_to_hashable_tuple, given))
    comparable_expected = set(map(helpers.covert_idx_to_hashable_tuple, expected))

    assert comparable_given == comparable_expected


@pytest.mark.unit
def test_slice_idx_generator_z2():
    """Test convert.slice_idx_generator at zoom level 2.

    The given shape (4305, 9791) breaks iterative schemes that don't properly
    seperate tiles. Was a bug.
    """
    shape = (4305, 9791)
    zoom = 2
    tile_size = 256
    given = convert.slice_idx_generator(shape, zoom, tile_size)
    expected = helpers.get_slice_idx_generator_solution(zoom)

    comparable_given = set(map(helpers.covert_idx_to_hashable_tuple, given))
    comparable_expected = set(map(helpers.covert_idx_to_hashable_tuple, expected))

    assert comparable_given == comparable_expected


@pytest.mark.unit
def test_slice_idx_generator_z3():
    """Test convert.slice_idx_generator at zoom level 3.

    The given shape (4305, 9791) breaks iterative schemes that don't properly
    seperate tiles. Was a bug.
    """
    shape = (4305, 9791)
    zoom = 3
    tile_size = 256
    given = convert.slice_idx_generator(shape, zoom, tile_size)
    expected = helpers.get_slice_idx_generator_solution(zoom)

    comparable_given = set(map(helpers.covert_idx_to_hashable_tuple, given))
    comparable_expected = set(map(helpers.covert_idx_to_hashable_tuple, expected))

    assert comparable_given == comparable_expected


@pytest.mark.unit
def test_slice_idx_generator_z4():
    """Test convert.slice_idx_generator at zoom level 4.

    The given shape (4305, 9791) breaks iterative schemes that don't properly
    seperate tiles. Was a bug.
    """
    shape = (4305, 9791)
    zoom = 4
    tile_size = 256
    given = convert.slice_idx_generator(shape, zoom, tile_size)
    expected = helpers.get_slice_idx_generator_solution(zoom)

    comparable_given = set(map(helpers.covert_idx_to_hashable_tuple, given))
    comparable_expected = set(map(helpers.covert_idx_to_hashable_tuple, expected))

    assert comparable_given == comparable_expected


@pytest.mark.unit
def test_slice_idx_generator_z5():
    """Test convert.slice_idx_generator at zoom level 5.

    The given shape (4305, 9791) breaks iterative schemes that don't properly
    seperate tiles. Was a bug.
    """
    shape = (4305, 9791)
    zoom = 5
    tile_size = 256
    given = convert.slice_idx_generator(shape, zoom, tile_size)
    expected = helpers.get_slice_idx_generator_solution(zoom)

    comparable_given = set(map(helpers.covert_idx_to_hashable_tuple, given))
    comparable_expected = set(map(helpers.covert_idx_to_hashable_tuple, expected))

    assert comparable_given == comparable_expected


@pytest.mark.unit
def test_balance_array_2d():
    """Test convert.balance_array"""

    in_shape = (10, 20)
    expected_shape = (32, 32)
    expected_num_nans = np.prod(expected_shape) - np.prod(in_shape)

    test_array = np.zeros(in_shape)

    out_array = convert.balance_array(test_array)

    assert out_array.shape == expected_shape
    assert np.isnan(out_array).sum() == expected_num_nans


@pytest.mark.unit
def test_balance_array_3d():
    """Test convert.balance_array"""

    in_shape = (10, 20, 3)
    expected_shape = (32, 32, 3)
    expected_num_nans = np.prod(expected_shape) - np.prod(in_shape)

    test_array = np.zeros(in_shape)

    out_array = convert.balance_array(test_array)

    assert out_array.shape == expected_shape
    assert np.isnan(out_array).sum() == expected_num_nans


@pytest.mark.unit
def test_get_array_fits():
    """Test convert.get_array"""

    helpers.setup()

    # make test array
    tmp = np.zeros((3, 3), dtype=np.float32)
    out_path = os.path.join(helpers.TEST_PATH, "test.fits")
    fits.PrimaryHDU(data=tmp).writeto(out_path)

    pads = [[0, 1], [0, 1]]
    expected_array = np.pad(tmp, pads, mode="constant", constant_values=np.nan)

    # get test array
    actual_array = convert.get_array(out_path)

    helpers.tear_down()

    np.testing.assert_equal(expected_array, actual_array)


@pytest.mark.unit
def test_get_array_png():
    """Test convert.get_array"""

    helpers.setup()

    # make test array
    expected_array = camera()
    out_path = os.path.join(helpers.TEST_PATH, "test.png")
    Image.fromarray(expected_array).save(out_path)

    # get test array
    actual_array = convert.get_array(out_path)

    helpers.tear_down()

    np.testing.assert_equal(expected_array, np.flipud(actual_array))


@pytest.mark.unit
def test_filter_on_extension_without_predicate():
    """Test convert.filter_on_extension without a predicate argument"""

    test_files = ["file_one.fits", "file_two.fits", "file_three.exclude"]

    extensions = ["fits"]
    expected_list = test_files[:-1]

    actual_list = convert.filter_on_extension(test_files, extensions)

    assert expected_list == actual_list


@pytest.mark.unit
def test_filter_on_extension_with_predicate():
    """Test convert.filter_on_extension with a predicate argument"""

    test_files = ["file_one.fits", "file_two.fits", "file_three.exclude"]

    extensions = ["fits"]
    expected_list = test_files[:1]
    predicate = lambda f: f == test_files[1]

    actual_list = convert.filter_on_extension(test_files, extensions, predicate)

    assert expected_list == actual_list


@pytest.mark.unit
def test_make_dirs():
    """Test convert.make_dirs"""

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

    convert.make_dirs(out_dir, 0, 2)

    dirs_exists = all(map(os.path.exists, expected_dirs))

    helpers.tear_down()

    assert dirs_exists


@pytest.mark.unit
def test_get_zoom_range():
    """Test convert.get_zoom_range"""

    in_shape = [10000, 10000]
    tile_size = [256, 256]

    expected_min = 0
    expected_max = 5

    actual_min, acutal_max = convert.get_zoom_range(in_shape, tile_size)

    assert expected_min == actual_min
    assert expected_max == acutal_max


@pytest.mark.unit
def test_get_total_tiles():
    """Test convert.get_total_tiles"""

    min_zoom, max_zoom = 0, 2
    expected_number = 21

    actual_number = convert.get_total_tiles(min_zoom, max_zoom)
    assert expected_number == actual_number


@pytest.mark.unit
def test_get_map_layer_name():
    """Test convert.get_map_layer_name"""

    test_file_name = "./test/test_file.png"
    expected_layer_name = "test_file"

    actual_layer_name = convert.get_map_layer_name(test_file_name)

    assert expected_layer_name == actual_layer_name


@pytest.mark.unit
def test_get_marker_file_name():
    """Test convert.get_marker_file_names"""

    test_file_name = "./test/test_file.cat"
    expected_marker_file_name = "test_file.cat.js"

    actual_marker_file_name = convert.get_marker_file_name(test_file_name)

    assert expected_marker_file_name == actual_marker_file_name


@pytest.mark.unit
def test_line_to_cols():
    """Test convert.line_to_cols"""

    line = "id ra dec test1 test2"
    catalog_delim = None
    expected_cols = line.split()

    actual_cols = convert.line_to_cols(catalog_delim, line)

    assert expected_cols == actual_cols


@pytest.mark.unit
def test_line_to_cols_with_hash():
    """Test convert.line_to_cols"""

    line = "# id ra dec test1 test2"
    catalog_delim = None
    expected_cols = line.split()[1:]

    actual_cols = convert.line_to_cols(catalog_delim, line)

    assert expected_cols == actual_cols


@pytest.mark.unit
def test_line_to_json_xy():
    """Test convert.line_to_json with x/y"""
    in_wcs = None
    catalog_delim = None
    columns = ["id", "x", "y", "col1", "col2"]
    dims = [1000, 1000]
    in_line = "1 10 20 abc 123"

    html_row = "<tr><td><b>{}:<b></td><td>{}</td></tr>"
    src_rows = list(
        map(lambda z: html_row.format(*z), zip(columns, in_line.strip().split()))
    )

    src_desc = "".join(
        [
            "<span style='text-decoration:underline; font-weight:bold'>Catalog Information</span>",
            "<br>",
            "<table>",
            *src_rows,
            "</table>",
        ]
    )

    expected_json = dict(x=10, y=20, catalog_id="1", desc=src_desc)

    actual_json = convert.line_to_json(None, columns, catalog_delim, dims, in_line)

    assert expected_json == actual_json


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:.*:astropy.io.fits.verify.VerifyWarning")
def test_line_to_json_ra_dec():
    """Test convert.line_to_json with ra/dec"""
    helpers.setup(with_data=True)

    wcs = WCS(fits.getheader(os.path.join(helpers.TEST_PATH, "test_image.fits")))

    columns = ["id", "ra", "dec", "col1", "col2"]
    dims = [738, 738]
    catalog_delim = None
    in_line = "1  53.18575  -27.898664  test_1 abc"

    html_row = "<tr><td><b>{}:<b></td><td>{}</td></tr>"

    src_rows = list(
        map(lambda z: html_row.format(*z), zip(columns, in_line.strip().split()))
    )

    src_desc = "".join(
        [
            "<span style='text-decoration:underline; font-weight:bold'>Catalog Information</span>",
            "<br>",
            "<table>",
            *src_rows,
            "</table>",
        ]
    )

    expected_json = dict(
        x=289.37867109328727, y=300.7526406693396, catalog_id="1", desc=src_desc
    )

    actual_json = convert.line_to_json(wcs, columns, catalog_delim, dims, in_line)
    print(actual_json)

    assert expected_json == actual_json


@pytest.mark.unit
def test_async_worker_completes():
    """Test convert.async_worker"""

    q = JoinableQueue()
    q.put((lambda v1, v2: None, ["v1", "v2"]))

    convert.async_worker(q)

    assert True  # if we make it here it works


@pytest.mark.unit
def test_make_tile_mpl():
    """Test convert.make_tile_mpl"""
    helpers.setup()

    out_dir = helpers.TEST_PATH
    test_arr = np.arange(100 * 100).reshape([100, 100])
    test_job = (0, 0, 0, slice(0, 100), slice(0, 100))
    vmin = test_arr.min()
    vmax = test_arr.max()

    os.makedirs(os.path.join(out_dir, "0/0/"))

    convert.make_tile_mpl(vmin, vmax, out_dir, test_arr, test_job)

    actual_img = np.array(Image.open(os.path.join(out_dir, "0/0/0.png")))

    expected_img = test_arr // 256

    np.array_equal(expected_img, actual_img)

    helpers.tear_down()


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:.*:astropy.io.fits.verify.VerifyWarning")
def test_catalog_to_markers_xy():
    """Test convert.catalog_to_markers using xy coords"""
    helpers.disbale_tqdm()
    helpers.setup(with_data=True)

    out_dir = helpers.TEST_PATH
    wcs_file = os.path.join(out_dir, "test_image.fits")
    header = fits.getheader(wcs_file)
    catalog_bounds = header["NAXIS2"], header["NAXIS1"]
    catalog_file = os.path.join(out_dir, "test_catalog_xy.cat")
    catalog_delim = None
    pbar_loc = 0

    convert.catalog_to_markers(
        wcs_file, out_dir, catalog_delim, catalog_bounds, catalog_file, pbar_loc
    )

    expected_json, expected_name = helpers.cat_to_json(
        os.path.join(out_dir, "expected_test_catalog_xy.cat.js")
    )
    actual_json, actual_name = helpers.cat_to_json(
        os.path.join(out_dir, "js", "test_catalog_xy.cat.js")
    )

    helpers.tear_down()
    helpers.enable_tqdm()

    assert expected_json == actual_json
    assert expected_name == actual_name


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:.*:astropy.io.fits.verify.VerifyWarning")
def test_catalog_to_markers_radec():
    """Test convert.catalog_to_markers using xy coords"""
    helpers.disbale_tqdm()
    helpers.setup(with_data=True)

    out_dir = helpers.TEST_PATH
    wcs_file = os.path.join(out_dir, "test_image.fits")
    header = fits.getheader(wcs_file)
    catalog_bounds = header["NAXIS2"], header["NAXIS1"]
    catalog_file = os.path.join(out_dir, "test_catalog_radec.cat")
    catalog_delim = None
    pbar_loc = 0

    convert.catalog_to_markers(
        wcs_file, out_dir, catalog_delim, catalog_bounds, catalog_file, pbar_loc
    )

    expected_json, expcted_name = helpers.cat_to_json(
        os.path.join(out_dir, "expected_test_catalog_radec.cat.js")
    )
    actual_json, actual_name = helpers.cat_to_json(
        os.path.join(out_dir, "js", "test_catalog_radec.cat.js")
    )

    helpers.tear_down()
    helpers.enable_tqdm()

    assert expected_json == actual_json
    assert expcted_name == actual_name


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:.*:astropy.io.fits.verify.VerifyWarning")
def test_catalog_to_markers_fails():
    """Test convert.catalog_to_markers using xy coords"""
    helpers.disbale_tqdm()
    helpers.setup(with_data=True)

    out_dir = helpers.TEST_PATH
    wcs_file = os.path.join(out_dir, "test_image.fits")
    header = fits.getheader(wcs_file)
    catalog_bounds = header["NAXIS2"], header["NAXIS1"]
    catalog_delim = None
    catalog_file = os.path.join(out_dir, "test_catalog_fails.cat")
    pbar_loc = 0

    with pytest.raises(ValueError) as excinfo:
        convert.catalog_to_markers(
            wcs_file, out_dir, catalog_delim, catalog_bounds, catalog_file, pbar_loc
        )

    helpers.tear_down()
    helpers.enable_tqdm()

    assert "is missing coordinate columns" in str(excinfo.value)


@pytest.mark.unit
def test_tile_img_pil_serial():
    """Test convert.tile_img"""
    helpers.disbale_tqdm()
    helpers.setup(with_data=True)

    out_dir = helpers.TEST_PATH
    test_image = os.path.join(out_dir, "test_tiling_image.jpg")
    pbar_loc = 0
    min_zoom = 0
    image_engine = convert.IMG_ENGINE_PIL

    convert.tile_img(
        test_image,
        pbar_loc,
        min_zoom=min_zoom,
        image_engine=image_engine,
        out_dir=out_dir,
    )

    expected_dir = os.path.join(out_dir, "expected_test_tiling_image_pil")
    actual_dir = os.path.join(out_dir, "test_tiling_image")

    dirs_match = helpers.compare_file_directories(expected_dir, actual_dir)

    helpers.tear_down()
    helpers.enable_tqdm()

    assert dirs_match


@pytest.mark.unit
def test_tile_img_mpl_serial():
    """Test convert.tile_img"""
    helpers.disbale_tqdm()
    helpers.setup(with_data=True)

    out_dir = helpers.TEST_PATH
    test_image = os.path.join(out_dir, "test_tiling_image.jpg")
    pbar_loc = 0
    min_zoom = 0
    image_engine = convert.IMG_ENGINE_MPL

    convert.tile_img(
        test_image,
        pbar_loc,
        min_zoom=min_zoom,
        image_engine=image_engine,
        out_dir=out_dir,
    )

    expected_dir = os.path.join(out_dir, "expected_test_tiling_image_mpl")
    actual_dir = os.path.join(out_dir, "test_tiling_image")

    dirs_match = helpers.compare_file_directories(expected_dir, actual_dir)

    helpers.tear_down()
    helpers.enable_tqdm()

    assert dirs_match


@pytest.mark.unit
def test_tile_img_pil_parallel():
    """Test convert.tile_img"""
    helpers.disbale_tqdm()
    helpers.setup(with_data=True)

    out_dir = helpers.TEST_PATH
    test_image = os.path.join(out_dir, "test_tiling_image.jpg")
    pbar_loc = 0
    min_zoom = 0
    image_engine = convert.IMG_ENGINE_PIL

    convert.tile_img(
        test_image,
        pbar_loc,
        min_zoom=min_zoom,
        image_engine=image_engine,
        out_dir=out_dir,
        mp_procs=2,
    )

    expected_dir = os.path.join(out_dir, "expected_test_tiling_image_pil")
    actual_dir = os.path.join(out_dir, "test_tiling_image")

    dirs_match = helpers.compare_file_directories(expected_dir, actual_dir)

    helpers.tear_down()
    helpers.enable_tqdm()

    assert dirs_match


@pytest.mark.unit
def test_tile_img_mpl_parallel():
    """Test convert.tile_img"""
    helpers.disbale_tqdm()
    helpers.setup(with_data=True)

    out_dir = helpers.TEST_PATH
    test_image = os.path.join(out_dir, "test_tiling_image.jpg")
    pbar_loc = 0
    min_zoom = 0
    image_engine = convert.IMG_ENGINE_MPL

    convert.tile_img(
        test_image,
        pbar_loc,
        min_zoom=min_zoom,
        image_engine=image_engine,
        out_dir=out_dir,
        mp_procs=2,
    )

    expected_dir = os.path.join(out_dir, "expected_test_tiling_image_mpl")
    actual_dir = os.path.join(out_dir, "test_tiling_image")

    dirs_match = helpers.compare_file_directories(expected_dir, actual_dir)

    helpers.tear_down()
    helpers.enable_tqdm()

    assert dirs_match


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore:.*:astropy.io.fits.verify.VerifyWarning")
def test_files_to_map():
    """Integration test for making files into map"""
    helpers.disbale_tqdm()
    helpers.setup(with_data=True)

    with_path = lambda f: os.path.join(helpers.TEST_PATH, f)
    out_dir = with_path("test_web")

    files = [with_path("test_tiling_image.jpg"), with_path("test_catalog_radec.cat")]

    convert.files_to_map(
        files, out_dir=out_dir, cat_wcs_fits_file=with_path("test_image.fits")
    )

    expected_dir = with_path("expected_test_web")
    actual_dir = with_path("test_web")

    dirs_match = helpers.compare_file_directories(expected_dir, actual_dir)

    helpers.tear_down()
    helpers.enable_tqdm()

    assert dirs_match


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore:.*:astropy.io.fits.verify.VerifyWarning")
def test_files_to_map_fails():
    """Integration test for making files into map"""
    helpers.disbale_tqdm()
    helpers.setup(with_data=True)

    with_path = lambda f: os.path.join(helpers.TEST_PATH, f)
    out_dir = with_path("test_web")

    files = [
        with_path("test_tiling_image.jpg"),
        with_path("test_catalog_radec.cat"),
        with_path("does_not_exist.txt"),
    ]

    with pytest.raises(FileNotFoundError):
        convert.files_to_map(
            files, out_dir=out_dir, cat_wcs_fits_file=with_path("test_image.fits")
        )

    helpers.tear_down()
    helpers.enable_tqdm()
