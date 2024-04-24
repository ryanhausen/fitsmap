# MIT License
# Copyright 2023 Ryan Hausen and contributors

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
import queue
import shutil
import sys

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
@pytest.mark.convert
def test_build_path():
    """Test the convert.build_path function"""
    z, y, x = 1, 2, 3
    out_dir = helpers.TEST_PATH
    img_name = convert.build_path(z, y, x, out_dir)

    expected_img_name = os.path.join(out_dir, str(z), str(y), f"{x}.png")

    expected_file_name_matches = expected_img_name == img_name

    assert expected_file_name_matches


@pytest.mark.unit
@pytest.mark.convert
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
@pytest.mark.convert
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
@pytest.mark.convert
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
@pytest.mark.convert
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
@pytest.mark.convert
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
@pytest.mark.convert
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
@pytest.mark.convert
def test_slice_idx_generator_raises():
    """Test convert.slice_idx_generator raises StopIteration.

    The given shape (4305, 9791) breaks iterative schemes that don't properly
    seperate tiles. Was a bug.
    """
    shape = (250, 250)
    zoom = 5
    tile_size = 256

    with pytest.raises(StopIteration) as excinfo:
        given = convert.slice_idx_generator(shape, zoom, tile_size)

    assert excinfo


@pytest.mark.unit
@pytest.mark.convert
def test_balance_array_2d():
    """Test convert.balance_array"""

    in_shape = (10, 20)
    expected_shape = (32, 32)
    expected_num_nans = np.prod(expected_shape) - np.prod(in_shape)

    test_array = np.zeros(in_shape)

    out_array = convert.balance_array(test_array)

    assert out_array.shape == expected_shape
    assert np.isnan(out_array[:]).sum() == expected_num_nans


@pytest.mark.unit
@pytest.mark.convert
def test_balance_array_3d():
    """Test convert.balance_array"""

    in_shape = (10, 20, 3)
    expected_shape = (32, 32, 3)
    expected_num_nans = np.prod(expected_shape) - np.prod(in_shape)

    test_array = np.zeros(in_shape)

    out_array = convert.balance_array(test_array)

    assert out_array.shape == expected_shape
    assert np.isnan(out_array[:]).sum() == expected_num_nans


@pytest.mark.unit
@pytest.mark.convert
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

    np.testing.assert_equal(expected_array, actual_array[:])


@pytest.mark.unit
@pytest.mark.convert
def test_get_array_fits_fails():
    """Test convert.get_array"""

    helpers.setup()

    # make test array
    tmp = np.zeros((3), dtype=np.float32)
    out_path = os.path.join(helpers.TEST_PATH, "test.fits")
    fits.PrimaryHDU(data=tmp).writeto(out_path)

    with pytest.raises(ValueError) as excinfo:
        convert.get_array(out_path)

    helpers.tear_down()

    assert "FitsMap only supports 2D" in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.convert
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

    np.testing.assert_equal(expected_array, np.flipud(actual_array.array))


@pytest.mark.unit
@pytest.mark.convert
def test_filter_on_extension_without_predicate():
    """Test convert.filter_on_extension without a predicate argument"""

    test_files = ["file_one.fits", "file_two.fits", "file_three.exclude"]

    extensions = ["fits"]
    expected_list = test_files[:-1]

    actual_list = convert.filter_on_extension(test_files, extensions)

    assert expected_list == actual_list


@pytest.mark.unit
@pytest.mark.convert
def test_filter_on_extension_with_predicate():
    """Test convert.filter_on_extension with a predicate argument"""

    test_files = ["file_one.fits", "file_two.fits", "file_three.exclude"]

    extensions = ["fits"]
    expected_list = test_files[:1]
    predicate = lambda f: f == test_files[1]

    actual_list = convert.filter_on_extension(test_files, extensions, predicate)

    assert expected_list == actual_list


@pytest.mark.unit
@pytest.mark.convert
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
@pytest.mark.convert
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
@pytest.mark.convert
def test_get_total_tiles():
    """Test convert.get_total_tiles"""

    min_zoom, max_zoom = 0, 2
    expected_number = 21

    actual_number = convert.get_total_tiles(min_zoom, max_zoom)
    assert expected_number == actual_number


@pytest.mark.unit
@pytest.mark.convert
def test_imread_default():
    """Test convert.imread_default() with valid path"""

    helpers.setup(with_data=True)

    test_file = os.path.join(helpers.TEST_PATH, "test_tiling_image.jpg")
    expected_array = np.flipud(Image.open(test_file))
    empty_array = np.zeros([256, 256])

    actual_array = convert.imread_default(test_file, empty_array)

    helpers.tear_down()

    np.testing.assert_equal(expected_array, actual_array)


@pytest.mark.unit
@pytest.mark.convert
def test_imread_default_invalid_path():
    """Test convert.imread_default() with valid path"""

    helpers.setup(with_data=True)

    test_file = os.path.join(helpers.TEST_PATH, "doesnt_exist.jpg")
    empty_array = np.zeros([256, 256])

    actual_array = convert.imread_default(test_file, empty_array)

    helpers.tear_down()

    np.testing.assert_equal(empty_array, actual_array)


@pytest.mark.unit
@pytest.mark.convert
def test_get_map_layer_name():
    """Test convert.get_map_layer_name"""

    test_file_name = "./test/test_file.png"
    expected_layer_name = "test_file"

    actual_layer_name = convert.get_map_layer_name(test_file_name)

    assert expected_layer_name == actual_layer_name


@pytest.mark.unit
@pytest.mark.convert
def test_get_marker_file_name():
    """Test convert.get_marker_file_names"""

    test_file_name = "./test/test_file.cat"
    expected_marker_file_name = "test_file.cat.js"

    actual_marker_file_name = convert.get_marker_file_name(test_file_name)

    assert expected_marker_file_name == actual_marker_file_name


@pytest.mark.unit
@pytest.mark.convert
def test_line_to_cols():
    """Test convert.line_to_cols"""

    line = ["ID", "RA", "dec", "test1", "test2"]

    actual_cols = convert.line_to_cols(line)
    expected_cols = line
    expected_cols[0] = "id"
    expected_cols[1] = "ra"

    assert expected_cols == actual_cols


@pytest.mark.unit
@pytest.mark.convert
def test_line_to_cols_with_hash():
    """Test convert.line_to_cols"""

    line = ["#", "ID", "RA", "dec", "test1", "test2"]

    actual_cols = convert.line_to_cols(line)
    expected_cols = line[1:]
    expected_cols[0] = "id"
    expected_cols[1] = "ra"

    assert expected_cols == actual_cols


@pytest.mark.unit
@pytest.mark.convert
def test_line_to_json_xy():
    """Test convert.line_to_json with x/y"""

    helpers.setup()

    in_wcs = None
    columns = ["id", "x", "y", "col1", "col2"]
    catalog_assets_path = os.path.join(helpers.TEST_PATH, "catalog_assets")
    os.mkdir(catalog_assets_path)
    in_line = ["1", "10", "20", "abc", "123"]

    expected_json = dict(
        geometry=dict(coordinates=[9.5, 19.5]),
        tags=dict(
            a=-1,
            b=-1,
            theta=-1,
            catalog_id="1",
            cat_path="catalog_assets",
        ),
    )

    actual_json = convert.line_to_json(
        in_wcs,
        columns,
        catalog_assets_path,
        in_line,
    )

    helpers.tear_down()

    assert expected_json == actual_json


@pytest.mark.unit
@pytest.mark.convert
@pytest.mark.filterwarnings("ignore:.*:astropy.io.fits.verify.VerifyWarning")
def test_line_to_json_ra_dec():
    """Test convert.line_to_json with ra/dec"""
    helpers.setup(with_data=True)

    in_wcs = WCS(fits.getheader(os.path.join(helpers.TEST_PATH, "test_image.fits")))

    columns = ["id", "ra", "dec", "col1", "col2"]
    catalog_assets_path = os.path.join(helpers.TEST_PATH, "catalog_assets")
    os.mkdir(catalog_assets_path)
    in_line = ["1", "53.18575", "-27.898664", "abc", "123"]

    expected_json = dict(
        geometry=dict(coordinates=[289.87867109328727, 301.2526406693396]),
        tags=dict(
            a=-1,
            b=-1,
            theta=-1,
            catalog_id="1",
            cat_path="catalog_assets",
        ),
    )

    actual_json = convert.line_to_json(
        in_wcs,
        columns,
        catalog_assets_path,
        in_line,
    )

    helpers.tear_down()

    np.testing.assert_allclose(
        expected_json["geometry"]["coordinates"],
        actual_json["geometry"]["coordinates"],
        atol=1e-6,
    )

    assert expected_json["tags"] == actual_json["tags"]


@pytest.mark.unit
@pytest.mark.convert
def test_tile_img_pil_serial():
    """Test convert.tile_img"""
    helpers.disbale_tqdm()
    helpers.setup(with_data=True)

    out_dir = helpers.TEST_PATH
    test_image = os.path.join(out_dir, "test_tiling_image.jpg")
    pbar_ref = [0, queue.Queue()]

    convert.tile_img(
        test_image,
        pbar_ref,
        out_dir=out_dir,
    )

    expected_dir = os.path.join(out_dir, "expected_test_tiling_image_pil")
    actual_dir = os.path.join(out_dir, "test_tiling_image")

    dirs_match = helpers.compare_file_directories(expected_dir, actual_dir)

    helpers.tear_down()
    helpers.enable_tqdm()

    assert dirs_match


@pytest.mark.unit
@pytest.mark.convert
def test_tile_img_pil_serial_png_from_tiff():
    """Test convert.tile_img using a converted TIFF->PNG image, has only 2 dims"""
    helpers.disbale_tqdm()
    helpers.setup(with_data=True)

    out_dir = helpers.TEST_PATH
    test_image = os.path.join(out_dir, "test_png_from_tiff.png")
    pbar_ref = [0, queue.Queue()]

    convert.tile_img(
        test_image,
        pbar_ref,
        out_dir=out_dir,
    )

    expected_dir = os.path.join(out_dir, "expected_test_png_from_tiff")
    actual_dir = os.path.join(out_dir, "test_png_from_tiff")

    dirs_match = helpers.compare_file_directories(expected_dir, actual_dir)

    helpers.tear_down()
    helpers.enable_tqdm()

    assert dirs_match


@pytest.mark.unit
@pytest.mark.convert
def test_tile_img_mpl_fits_serial():
    """Test convmax_percentert.tile_img"""
    helpers.disbale_tqdm()
    helpers.setup(with_data=True)

    out_dir = helpers.TEST_PATH
    test_image = os.path.join(out_dir, "test_img_for_map.fits")
    pbar_ref = [0, queue.Queue()]

    convert.tile_img(
        test_image,
        pbar_ref,
        out_dir=out_dir,
        norm_kwargs=dict(stretch="log", max_percent=99.9),
    )

    expected_dir = os.path.join(out_dir, "expected_test_img_for_map")
    actual_dir = os.path.join(out_dir, "test_img_for_map")

    dirs_match = helpers.compare_file_directories(expected_dir, actual_dir)

    helpers.tear_down()
    helpers.enable_tqdm()

    assert dirs_match


@pytest.mark.unit
@pytest.mark.convert
def test_tile_img_mpl_fits_serial_with_fname_kwargs():
    """Test convmax_percentert.tile_img"""
    helpers.disbale_tqdm()
    helpers.setup(with_data=True)

    out_dir = helpers.TEST_PATH
    test_image = os.path.join(out_dir, "test_img_for_map.fits")
    pbar_ref = [0, queue.Queue()]

    convert.tile_img(
        test_image,
        pbar_ref,
        out_dir=out_dir,
        norm_kwargs={
            "test_img_for_map.fits": dict(stretch="log", max_percent=99.9),
        },
    )

    expected_dir = os.path.join(out_dir, "expected_test_img_for_map")
    actual_dir = os.path.join(out_dir, "test_img_for_map")

    dirs_match = helpers.compare_file_directories(expected_dir, actual_dir)

    helpers.tear_down()
    helpers.enable_tqdm()

    assert dirs_match


def test_simplify_mixed_ws():
    """Test convert._simplify_mixed_ws"""
    helpers.disbale_tqdm()
    helpers.setup(with_data=True)

    test_lines = [
        "a b   c\n",
        "test\tdata stuff\n",
        "to     test\tstuff\n",
    ]

    out_file = os.path.join(helpers.DATA_DIR, "test.cat")
    with open(out_file, "w") as f:
        f.writelines(test_lines)

    convert._simplify_mixed_ws(out_file)

    expected_test_lines = [
        "a b c\n",
        "test data stuff\n",
        "to test stuff\n",
    ]

    with open(out_file, "r") as f:
        actual_lines = f.readlines()

    helpers.tear_down()
    helpers.enable_tqdm()

    assert expected_test_lines == actual_lines


@pytest.mark.unit
@pytest.mark.convert
def test_tile_img_pil_parallel():
    """Test convert.tile_img"""
    helpers.disbale_tqdm()
    helpers.setup(with_data=True)

    out_dir = helpers.TEST_PATH
    test_image = os.path.join(out_dir, "test_tiling_image.jpg")
    pbar_ref = [0, queue.Queue()]

    convert.tile_img(
        test_image,
        pbar_ref,
        out_dir=out_dir,
        mp_procs=2,
    )

    expected_dir = os.path.join(out_dir, "expected_test_tiling_image_pil")
    actual_dir = os.path.join(out_dir, "test_tiling_image")

    dirs_match = helpers.compare_file_directories(expected_dir, actual_dir)

    helpers.tear_down(include_ray=True)
    helpers.enable_tqdm()

    assert dirs_match


@pytest.mark.unit
@pytest.mark.convert
def test_tile_img_mpl_parallel():
    """Test convert.tile_img"""
    helpers.disbale_tqdm()
    helpers.setup(with_data=True)

    out_dir = helpers.TEST_PATH
    test_image = os.path.join(out_dir, "test_img_for_map.fits")
    pbar_ref = [0, queue.Queue()]

    convert.tile_img(
        test_image,
        pbar_ref,
        out_dir=out_dir,
        mp_procs=2,
        norm_kwargs=dict(stretch="log", max_percent=99.9),
    )

    expected_dir = os.path.join(out_dir, "expected_test_img_for_map")
    actual_dir = os.path.join(out_dir, "test_img_for_map")

    dirs_match = helpers.compare_file_directories(expected_dir, actual_dir)

    helpers.tear_down(include_ray=True)
    helpers.enable_tqdm()

    assert dirs_match


@pytest.mark.unit
@pytest.mark.convert
@pytest.mark.skipif(
    condition=not sys.platform.startswith("linux"),
    reason="temp fix, need osx/windows artififacts for cbor/pbf files",
)
def test_version_not_hard_coded():
    """Tests that the version in the testing artifacts is not hard coded"""

    helpers.disbale_tqdm()
    helpers.setup(with_data=True)

    file = "index.html"
    dirs = [
        "expected_test_web",
        "expected_test_web_ellipse",
        "expected_test_web_no_marker",
    ]

    tests = {}
    for d in dirs:
        with open(os.path.join(helpers.TEST_PATH, d, file), "r") as f:
            text = f.read()
            tests[d] = "vVERSION" in text

    helpers.tear_down()
    helpers.enable_tqdm()
    failed = [d for d, v in tests.items() if v == False]
    assert len(failed) == 0, "VERSION not found in {}, likely hardcoded".format(failed)


@pytest.mark.integration
@pytest.mark.convert
@pytest.mark.skipif(
    condition=not sys.platform.startswith("linux"),
    reason="temp fix, need osx/windows artififacts for cbor/pbf files",
)
@pytest.mark.filterwarnings("ignore:.*:astropy.io.fits.verify.VerifyWarning")
def test_files_to_map():
    """Integration test for making files into map"""
    helpers.disbale_tqdm()
    helpers.setup(with_data=True)

    with_path = lambda f: os.path.join(helpers.TEST_PATH, f)
    out_dir = with_path("test_web")

    files = [with_path("test_tiling_image.jpg"), with_path("test_catalog_radec.cat")]

    convert.files_to_map(
        files,
        out_dir=out_dir,
        cat_wcs_fits_file=with_path("test_image.fits"),
        catalog_delim=" ",
    )

    expected_dir = with_path("expected_test_web")

    # inject current version in to test_index.html
    version = helpers.get_version()
    raw_path = os.path.join(expected_dir, "index.html")
    with open(raw_path, "r") as f:
        converted = list(map(lambda l: l.replace("VERSION", version), f.readlines()))

    with open(raw_path, "w") as f:
        f.writelines(converted)

    actual_dir = with_path("test_web")

    dirs_match = helpers.compare_file_directories(expected_dir, actual_dir)

    helpers.tear_down(include_ray=True)
    helpers.enable_tqdm()

    assert dirs_match


@pytest.mark.integration
@pytest.mark.convert
@pytest.mark.skipif(
    condition=not sys.platform.startswith("linux"),
    reason="temp fix, need osx/windows artififacts for cbor/pbf files",
)
@pytest.mark.filterwarnings("ignore:.*:astropy.io.fits.verify.VerifyWarning")
def test_files_to_map_ellipse_markers():
    """Integration test for making files into map"""
    helpers.disbale_tqdm()
    helpers.setup(with_data=True)

    with_path = lambda f: os.path.join(helpers.TEST_PATH, f)
    out_dir = with_path("test_web")

    files = [
        with_path("test_tiling_image.jpg"),
        with_path("test_catalog_xy_ellipse.cat"),
    ]

    convert.files_to_map(
        files,
        out_dir=out_dir,
        catalog_delim=" ",
    )

    expected_dir = with_path("expected_test_web_ellipse")

    # inject current version in to test_index.html
    version = helpers.get_version()
    raw_path = os.path.join(expected_dir, "index.html")
    with open(raw_path, "r") as f:
        converted = list(map(lambda l: l.replace("VERSION", version), f.readlines()))

    with open(raw_path, "w") as f:
        f.writelines(converted)

    actual_dir = with_path("test_web")

    dirs_match = helpers.compare_file_directories(expected_dir, actual_dir)

    helpers.tear_down(include_ray=True)
    helpers.enable_tqdm()

    assert dirs_match


@pytest.mark.integration
@pytest.mark.convert
@pytest.mark.filterwarnings("ignore:.*:astropy.io.fits.verify.VerifyWarning")
def test_files_to_map_fails_file_not_found():
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

    with pytest.raises(AssertionError):
        convert.files_to_map(
            files, out_dir=out_dir, cat_wcs_fits_file=with_path("test_image.fits")
        )

    helpers.tear_down()
    helpers.enable_tqdm()


@pytest.mark.integration
@pytest.mark.convert
@pytest.mark.filterwarnings("ignore:.*:astropy.io.fits.verify.VerifyWarning")
def test_dir_to_map_fails_no_files():
    """Integration test for making files into map"""
    helpers.disbale_tqdm()
    helpers.setup(with_data=True)

    with_path = lambda f: os.path.join(helpers.TEST_PATH, f)
    out_dir = with_path("test_web")
    in_dir = with_path("test_web_in")
    if not os.path.exists(in_dir):
        os.mkdir(in_dir)

    with pytest.raises(AssertionError):
        convert.dir_to_map(
            in_dir, out_dir=out_dir, cat_wcs_fits_file=with_path("test_image.fits")
        )

    helpers.tear_down()
    helpers.enable_tqdm()


@pytest.mark.integration
@pytest.mark.convert
@pytest.mark.skipif(
    condition=not sys.platform.startswith("linux"),
    reason="temp fix, need osx/windows artififacts for cbor/pbf files",
)
@pytest.mark.filterwarnings("ignore:.*:astropy.io.fits.verify.VerifyWarning")
def test_dir_to_map():
    """Integration test for making files into map"""
    helpers.disbale_tqdm()
    helpers.setup(with_data=True)

    with_path = lambda f: os.path.join(helpers.TEST_PATH, f)
    out_dir = with_path("test_web")
    in_dir = with_path("test_web_in")
    if not os.path.exists(in_dir):
        os.mkdir(in_dir)

    files = [
        "test_tiling_image.jpg",
        "test_catalog_radec.cat",
    ]

    for f in files:
        shutil.copy(with_path(f), os.path.join(in_dir, f))

    expected_dir = with_path("expected_test_web")

    # inject current version in to test_index.html
    version = helpers.get_version()
    raw_path = os.path.join(expected_dir, "index.html")
    with open(raw_path, "r") as f:
        converted = list(map(lambda l: l.replace("VERSION", version), f.readlines()))

    with open(raw_path, "w") as f:
        f.writelines(converted)

    convert.dir_to_map(
        in_dir,
        out_dir=out_dir,
        catalog_delim=" ",
        cat_wcs_fits_file=with_path("test_image.fits"),
    )

    actual_dir = out_dir

    dirs_match = helpers.compare_file_directories(expected_dir, actual_dir)

    helpers.tear_down(include_ray=True)
    helpers.enable_tqdm()

    assert dirs_match


@pytest.mark.integration
@pytest.mark.convert
@pytest.mark.filterwarnings("ignore:.*:astropy.io.fits.verify.VerifyWarning")
def test_dir_to_map_no_markers():
    """Integration test for making files into map"""
    helpers.disbale_tqdm()
    helpers.setup(with_data=True)

    with_path = lambda f: os.path.join(helpers.TEST_PATH, f)
    out_dir = with_path("test_web")
    in_dir = with_path("test_web_in")
    if not os.path.exists(in_dir):
        os.mkdir(in_dir)

    files = [
        "test_tiling_image.jpg",
    ]

    for f in files:
        shutil.copy(with_path(f), os.path.join(in_dir, f))

    expected_dir = with_path("expected_test_web_no_marker")

    # inject current version in to test_index.html
    version = helpers.get_version()
    raw_path = os.path.join(expected_dir, "index.html")
    with open(raw_path, "r") as f:
        converted = list(map(lambda l: l.replace("VERSION", version), f.readlines()))

    with open(raw_path, "w") as f:
        f.writelines(converted)

    convert.dir_to_map(
        in_dir,
        out_dir=out_dir,
    )

    actual_dir = out_dir

    dirs_match = helpers.compare_file_directories(expected_dir, actual_dir)

    helpers.tear_down(include_ray=True)
    helpers.enable_tqdm()

    assert dirs_match
