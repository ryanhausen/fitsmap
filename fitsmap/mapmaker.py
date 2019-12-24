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

"""Converts image files and catalogs into a leafletJS map."""

import json
import os
import shutil
import sys
from functools import partial, reduce
from itertools import chain, count, filterfalse, product, repeat
from multiprocessing import JoinableQueue, Pool, Process
from queue import Empty
from typing import Callable, Iterable, List, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sharedmem
from astropy.io import fits
from astropy.wcs import WCS
from imageio import imread
from PIL import Image
from tqdm import tqdm

from fitsmap.web_map import Map

# https://github.com/zimeon/iiif/issues/11#issuecomment-131129062
Image.MAX_IMAGE_PIXELS = sys.maxsize


Shape = Tuple[int, int]

IMG_FORMATS = ["fits", "jpg", "png"]
CAT_FORMAT = ["cat"]
IMG_ENGINE_PIL = "PIL"
IMG_ENGINE_MPL = "MPL"
MPL_CMAP = "gray"

# MPL SINGLETON ENGINE =========================================================
mpl_f, mpl_img, mpl_alpha_f = None, None, None
# ===============================================================================


def build_path(z, y, x, out_dir) -> str:
    """Maps zoom and coordinate location to a subdir in ``out_dir``

    Args:
        z (int): The zoom level for the tiles
        y (int): The zoom level for the tiles
        x (int): The zoom level for the tiles
        out_dir (str): The root directory the tiles are saved in
    Returns:
        The str path to save the tile in
    """
    z, y, x = str(z), str(y), str(x)
    z_dir = os.path.join(out_dir, z)
    y_dir = os.path.join(z_dir, y)

    img_path = os.path.join(y_dir, "{}.png".format(x))

    return img_path


def slice_idx_generator(
    shape: Tuple[int, int], zoom: int
) -> Iterable[Tuple[int, int, int, slice, slice]]:
    """Generates the tile coordinates and respective slices given a shape and zoom

    Args:
        shape (Tuple[int, int]): The shape of the array being tiled
        zoom (int): The zoom level for the tiles
    Returns:
        An iterable of tuples containing the following:
        (zoom, y_coordinate, x_coordinate, dim0_slice, dim1_slice)
    """
    default_splits = int((4 ** zoom) ** 0.5)

    num_rows, num_cols = shape[:2]
    img_ratio = int(max(shape[:2]) / min(shape[:2]))

    if img_ratio == 1:  # square
        num_splits_rows = num_splits_cols = default_splits
    else:  # rectangular
        short_splits = int(2 ** (zoom - 1))
        num_splits_rows = (
            short_splits if num_rows < num_cols else img_ratio * short_splits
        )
        num_splits_cols = (
            short_splits if num_rows > num_cols else img_ratio * short_splits
        )

    def split(vals):
        x0, x2 = vals
        x1 = x0 + ((x2-x0) // 2)
        return [(x0, x1), (x1, x2)]

    # split = lambda vals: [(vals[0], vals[1]//2), (vals[1]//2, vals[1])]
    split_collection = lambda collection: map(split, collection)
    split_reduce = lambda x, y: split_collection(chain.from_iterable(x))

    rows_split = list(reduce(split_reduce, repeat(None, zoom), [[(0, shape[0])]]))
    columns_split = list(reduce(split_reduce, repeat(None, zoom), [[(0, shape[1])]]))

    rows = zip(range(num_splits_rows - 1, -1, -1),  chain.from_iterable(rows_split))
    cols = enumerate(chain.from_iterable(columns_split))

    rows_cols = product(rows, cols)

    def transform_iteration(row_col):
        ((y, (start_y, end_y)), (x, (start_x, end_x))) = row_col
        return (zoom, y, x, slice(start_y, end_y), slice(start_x, end_x))

    return map(transform_iteration, rows_cols)


def balance_array(array: np.ndarray) -> np.ndarray:
    """Pads input array with zeros so that the long side is a multiple of the short side.

    Args:
        array (np.ndarray): array to balance
    Returns:
        a balanced version of ``array``
    """

    dim0, dim1 = array.shape[0], array.shape[1]

    # pad_dim0 = (dim1 - (dim0 % dim1 % dim0)) % dim1
    # pad_dim1 = (dim0 - (dim1 % dim0 % dim1)) % dim0
    pad_dim0 = max(dim1 - dim0, 0)
    pad_dim1 = max(dim0 - dim1, 0)

    if len(array.shape) == 3:
        padding = [[0, pad_dim0], [0, pad_dim1], [0, 0]]
    else:
        padding = [[0, pad_dim0], [0, pad_dim1]]

    return np.pad(
        array.astype(np.float32), padding, mode="constant", constant_values=np.nan
    )


def get_array(file_location: str) -> np.ndarray:
    """Opens the array at ``file_location`` can be an image or a fits file

    Args:
        file_location (str): the path to the image
    Returns:
        A numpy array representing the image.
    """

    _, ext = os.path.splitext(file_location)

    if ext == ".fits":
        array = fits.getdata(file_location)
        shape = array.shape
        if len(shape) > 2:
            raise ValueError("FitsMap only supports 2D FITS files.")
    else:
        array = np.flipud(imread(file_location))
        # array = imread(file_location)

        if len(array.shape) == 3:
            shape = array.shape[:-1]
        elif len(array.shape) == 2:
            shape = array.shape
        else:
            raise ValueError("FitsMap only supports 2D and 3D images.")

    if shape[0] != shape[1]:
        return balance_array(array)
    else:
        return array


def filter_on_extension(
    files: List[str], extensions: List[str], exclude_predicate: Callable = None
) -> List[str]:
    """Filters out files from ``files`` based on ``extensions`` and ``exclude_predicate``

    Args:
        files (List[str]): A list of file paths to be filtered
        extensions (List[str]): List of extensions to filter ``files`` on
        exclude_predicate (Callable): A function that accepts a single str as
                                      input and returns a True if the file
                                      should be excluded, and False if it should
                                      be included
    Returns:
        A list of files which have an extension thats in ``extensions`` and
        for which `exclude_predicate(file)==False``
    """

    neg_predicate = exclude_predicate if exclude_predicate else lambda x: False

    return list(
        filter(
            lambda s: (os.path.splitext(s)[1][1:] in extensions)
            and not neg_predicate(s),
            files,
        )
    )


def make_dirs(
    out_dir: str, min_zoom: int, max_zoom: int, shape: Tuple[int, int]
) -> None:
    """Builds the directory tree for storing image tiles.

    Args:
        out_dir (str): The root directory to generate the tree in
        min_zoom (int): The minimum zoom level the image will be tiled at
        max_zoom (int): The maximum zoom level the image will be tiled at
    Returns:
        None
    """

    num_rows, num_cols = shape[:2]
    img_ratio = int(max(shape[:2]) / min(shape[:2]))

    if img_ratio == 1:
        row_count = lambda z: int(np.sqrt(4 ** z))
    else:
        coefficient = 1 if (num_rows < num_cols) else img_ratio
        row_count = lambda z: int(2 ** (z - 1)) * coefficient

    def sub_dir(f):
        try:
            os.makedirs(os.path.join(out_dir, f))
        except FileExistsError:
            pass

    def build_z_ys(z, ys):
        list(map(lambda y: sub_dir(f"{z}/{y}"), ys))

    def build_zs(z):
        ys = range(row_count(z))
        build_z_ys(z, ys)

    zs = range(min_zoom, max_zoom + 1)
    list(map(build_zs, zs))


def get_zoom_range(
    shape: Tuple[int, int], tile_size: Tuple[int, int]
) -> Tuple[int, int]:
    """Returns the supported native zoom range for an give image size and tile size.

    Args:
        shape (Tuple[int, int]): The shape that is going to be tiled
        tile_size (Tuple[int, int]): The size of the image tiles
    Returns:
        A tuple containing the (minimum zoom level, maximum zoom level)
    """
    long_side = max(shape[:2])
    short_side = min(shape[:2])

    max_zoom = int(np.log2(long_side / tile_size[0]))
    min_zoom = int(np.ceil(np.log2(long_side / short_side)))

    return min_zoom, max_zoom


def get_total_tiles(shape: Tuple[int, int], min_zoom: int, max_zoom: int) -> int:
    """Returns the total number of tiles that will be generated from an image.

    Args:
        shape (Tuple[int, int]): The shape that is going to be tiled
        min_zoom (int): The minimum zoom level te image will be tiled at
        max_zoom (int): The maximum zoom level te image will be tiled at
    Returns:
        The total number of tiles that will be generated
    """
    img_ratio = int(max(shape[:2]) / min(shape[:2]))

    if img_ratio == 1:
        return int(sum([4 ** i for i in range(min_zoom, max_zoom + 1)]))
    else:
        return int(
            sum(
                [
                    (4 ** i) / 2 + ((4 ** i) / 4 * (img_ratio - 2))
                    for i in range(min_zoom, max_zoom + 1)
                ]
            )
        )


def make_tile(
    array: np.ndarray,
    vmin: float,
    vmax: float,
    out_dir: str,
    img_engine: str,
    job: Tuple[int, int, int, slice, slice],
) -> None:
    """Extracts a tile from ``array`` and saves it at the proper place in ``out_dir``.

    Args:
        array (np.ndarray): Array to extract a slice from
        vmin (float): The global minimum used to scale local values when
                      using the ``IMAGE_ENGINE_MPL`` image_engine.
        vmax (float): The global maximum used to scale local values when
                      using the ``IMAGE_ENGINE_MPL`` image_engine.
        out_dir (str): The directory to save tile in
        img_engine (str): Method to convert array tile to an image. Can be one
                          of mapmaker.IMAGE_ENGINE_PIL (using pillow(PIL)) or
                          of mapmaker.IMAGE_ENGINE_MPL (using matplotlib)
        job (Tuple[int, int, int, slice, slice]): A tuple containing z, y, x,
                                                  dim0_slices, dim1_slices. Where
                                                  (z, y, x) define the zoom and
                                                  the coordinates, and (dim0_slices,
                                                  and dim1_slices) are slice
                                                  objects that extract the tile.

    Returns:
        None
    """
    z, y, x, slice_ys, slice_xs = job

    img_path = build_path(z, y, x, out_dir)

    if img_engine == IMG_ENGINE_MPL:
        global mpl_f
        global mpl_img
        global mpl_alpha_f
        if mpl_f:
            mpl_img.set_data(mpl_alpha_f(array[slice_ys, slice_xs]))
            mpl_f.savefig(
                img_path,
                dpi=256,
                bbox_inches=0,
                interpolation="nearest",
                facecolor=(0, 0, 0, 0),
            )
        else:
            if len(array.shape) == 2:
                cmap = mpl.cm.get_cmap(MPL_CMAP)
                cmap.set_bad(color=(0, 0, 0, 0))

                img_kwargs = dict(
                    origin="lower",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                )
                mpl_alpha_f = lambda arr: arr
            else:
                img_kwargs = dict(interpolation="nearest", origin="lower")

                def adjust_pixels(arr):
                    img = arr.copy()
                    if img.shape[2] == 3:
                        img = np.concatenate(
                            (
                                img,
                                np.ones(list(img.shape[:-1]) + [1], dtype=np.float32)
                                * 255,
                            ),
                            axis=2,
                        )

                    ys, xs = np.where(np.isnan(img[:, :, 0]))
                    img[ys, xs, :] = np.array([0, 0, 0, 0], dtype=np.float32)

                    return img.astype(np.uint8)

                mpl_alpha_f = lambda arr: adjust_pixels(arr)

            mpl_f = plt.figure(dpi=256)
            mpl_f.set_size_inches([256 / 256, 256 / 256])
            mpl_img = plt.imshow(mpl_alpha_f(array[slice_ys, slice_xs]), **img_kwargs)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.axis("off")
            mpl_f.savefig(
                img_path,
                dpi=256,
                bbox_inches=0,
                interpolation="nearest",
                facecolor=(0, 0, 0, 0),
            )
    else:
        arr = array[slice_ys, slice_xs].copy()
        if len(arr.shape) < 3:
            arr = np.dstack([arr, arr, arr, np.ones_like(arr) * 255])
        elif arr.shape[2] == 3:
            arr = np.concatenate(
                (arr, np.ones(list(arr.shape[:-1]) + [1], dtype=np.float32) * 255),
                axis=2,
            )

        ys, xs = np.where(np.isnan(arr[:, :, 0]))
        arr[ys, xs, :] = np.array([0, 0, 0, 0], dtype=np.float32)
        img = Image.fromarray(np.flipud(arr).astype(np.uint8))
        del arr
        # img = Image.fromarray(arr.astype(np.uint8))

        img.thumbnail([256, 256], Image.LANCZOS)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        img.save(img_path, "PNG")
        del img


def tile_img(
    file_location: str,
    pbar_loc: int,
    tile_size: Shape = [256, 256],
    zoom: int = None,
    image_engine: str = IMG_ENGINE_PIL,
    out_dir: str = ".",
    mp_procs: int = 0,
) -> None:
    """Extracts tiles from the array at ``file_location``.

    Args:
        file_location (str): The file location of the image to tile
        pbar_loc (int): The index of the location of to print the tqdm bar
        tile_size (Tuple[int, int]): The pixel size of the tiles in the map
        zoom (int): The maximum zoom to create tiles for. If not provided the
                    value will be set to floor(log_2(img_height / tile_height))
        img_engine (str): Method to convert array tile to an image. Can be one
                          of mapmaker.IMAGE_ENGINE_PIL (using pillow(PIL)) or
                          of mapmaker.IMAGE_ENGINE_MPL (using matplotlib)
        out_dir (str): The root directory to save the tiles in
        mp_procs (int): The number of multiprocessing processes to use for
                        generating tiles.

    Returns:
        None
    """
    # reset mpl vars just in case they have been set by another img
    global mpl_f
    global mpl_img
    if mpl_f:
        mpl_f = None
        mpl_img = None

    # get image
    array = get_array(file_location)
    arr_min, arr_max = np.nanmin(array), np.nanmax(array)

    min_zoom, max_zoom = get_zoom_range(array.shape, tile_size)
    max_zoom = zoom if zoom else max_zoom

    # build directory structure
    name = get_map_layer_name(file_location)
    tile_dir = os.path.join(out_dir, name)
    if name not in os.listdir(out_dir):
        os.mkdir(tile_dir)

    make_dirs(tile_dir, min_zoom, max_zoom, array.shape)

    tile_params = chain.from_iterable(
        [slice_idx_generator(array.shape, i) for i in range(min_zoom, max_zoom + 1)]
    )

    # tile the image
    total_tiles = get_total_tiles(array.shape, min_zoom, max_zoom)

    if mp_procs:
        mp_array = sharedmem.empty_like(array)
        mp_array[:] = array[:]
        work = partial(make_tile, mp_array, arr_min, arr_max, tile_dir, image_engine)
        with Pool(mp_procs) as p:
            any(
                p.imap_unordered(
                    work,
                    tqdm(
                        tile_params,
                        desc="Converting " + name,
                        position=pbar_loc,
                        total=total_tiles,
                        unit="tile",
                    ),
                )
            )
    else:
        work = partial(make_tile, array, arr_min, arr_max, tile_dir, image_engine)
        any(
            map(
                work,
                tqdm(
                    tile_params,
                    desc="Converting " + name,
                    total=total_tiles,
                    unit="tile",
                    position=pbar_loc,
                ),
            )
        )


def get_map_layer_name(file_location: str) -> str:
    """Tranforms a ``file_location`` into the javascript layer name.

    Args:
        file_location (str): The file location to convert

    Returns:
        The javascript name that will be used in the HTML map
    """
    _, fname = os.path.split(file_location)
    name = os.path.splitext(fname)[0].replace(".", "_").replace("-", "_")
    return name


def get_marker_file_names(file_location: str):
    """Tranforms a ``file_location`` into the javascript marker file name.

    Args:
        file_location (str): The file location to convert

    Returns:
        The javascript name that will be used in the HTML map
    """
    return os.path.split(file_location)[1] + ".js"


def line_to_cols(raw_line: str):
    """Transform a raw text line of column names into a list of column names

    Args:
        raw_line (str): String from textfile

    Returns:
        A list of the column names in order
    """

    change_case = ["RA", "DEC", "Ra", "Dec", "X", "Y"]

    # make ra and dec lowercase for ease of access
    raw_cols = list(
        map(lambda s: s.lower() if s in change_case else s, raw_line.strip().split())
    )

    # if header line starts with a '#' exclude it
    if raw_cols[0] == "#":
        return raw_cols[1:]
    else:
        return raw_cols


def line_to_json(wcs: WCS, columns: List[str], max_dim: Tuple[int, int], src_line: str):
    """Transform a raw text line attribute values into a JSON marker

    Args:
        raw_line (str): String from the marker file

    Returns:
        A list of the column names in order
    """
    src_vals = src_line.strip().split()

    src_id = str(src_vals[columns.index("id")])
    if "x" in columns and "y" in columns:
        img_x = float(src_vals[columns.index("x")])
        img_y = float(src_vals[columns.index("y")])
    else:
        ra = float(src_vals[columns.index("ra")])
        dec = float(src_vals[columns.index("dec")])

        [[img_x, img_y]] = wcs.wcs_world2pix([[ra, dec]], 0)

    x = img_x / max_dim[1] * 256
    y = img_y / max_dim[0] * 256 - 256

    html_row = "<tr><td><b>{}:<b></td><td>{}</td></tr>"
    src_rows = list(map(lambda z: html_row.format(*z), zip(columns, src_vals)))

    src_desc = "".join(
        [
            "<span style='text-decoration:underline; font-weight:bold'>Catalog Information</span>",
            "<br>",
            "<table>",
            *src_rows,
            "</table>",
        ]
    )

    return dict(x=x, y=y, catalog_id=src_id, desc=src_desc)


def catalog_to_markers(
    wcs_file: str, out_dir: str, catalog_file: str, pbar_loc: int,
) -> None:
    """Transform ``catalog_file`` into a json collection for mapping

    Args:
        wcs_file (str): path to a FITS file to covert (ra, dec) to (x, y)
        out_dir (str): path to save the json collection in
        catalog_file (str): path to catalog file
        pbar_loc (int): the index to draw the tqdm bar in

    Returns:
        None
    """
    wcs = WCS(wcs_file)

    f = open(catalog_file, "r")

    columns = line_to_cols(next(f))

    ra_dec_coords = "ra" in columns and "dec" in columns
    x_y_coords = "x" in columns and "y" in columns

    if (not ra_dec_coords and not x_y_coords) or "id" not in columns:
        err_msg = " ".join(
            [
                catalog_file + " is missing coordinate columns (ra/dec, xy),",
                "an 'id' column, or all of the above",
            ]
        )
        raise ValueError(err_msg)

    header = fits.getheader(wcs_file)

    dim0, dim1 = header["NAXIS2"], header["NAXIS1"]

    # pad_dim0 = (dim1 - (dim0 % dim1 % dim0)) % dim1
    # pad_dim1 = (dim0 - (dim1 % dim0 % dim1)) % dim0
    pad_dim0 = max(dim1 - dim0, 0)
    pad_dim1 = max(dim0 - dim1, 0)

    line_func = partial(line_to_json, wcs, columns, [dim0 + pad_dim0, dim1 + pad_dim1])

    cat_file = os.path.split(catalog_file)[1] + ".js"

    if "js" not in os.listdir(out_dir):
        os.mkdir(os.path.join(out_dir, "js"))

    if "css" not in os.listdir(out_dir):
        os.mkdir(os.path.join(out_dir, "css"))

    json_markers_file = os.path.join(out_dir, "js", cat_file)
    with open(json_markers_file, "w") as j:
        j.write("var " + cat_file.replace(".cat.js", "") + " = ")
        json.dump(
            list(
                map(
                    line_func, tqdm(f, position=pbar_loc, desc="Converting " + cat_file)
                )
            ),
            j,
            indent=2,
        )
        j.write(";")
    f.close()


def async_worker(q: JoinableQueue):
    """Function for async task processesing.

    Args:
        q (JoinableQueue): Queue to retrieve tasks (func, args) from

    Returns:
        None
    """
    BLOCK = True
    TIMEOUT = 5

    while True:
        try:
            f, args = q.get(BLOCK, TIMEOUT)
            f(*args)
            q.task_done()
        except Empty:
            break


def files_to_map(
    files: List[str],
    out_dir: str = ".",
    zoom: int = None,
    title: str = "FitsMap",
    task_procs: int = 0,
    procs_per_task: int = 0,
    cat_wcs_fits_file: str = None,
    tile_size: Tuple[int, int] = [256, 256],
    image_engine: str = IMG_ENGINE_PIL,
):
    """Converts a list of files into a LeafletJS map.

    Args:
        files (List[str]): List of files to convert into a map, can include image
                           files (.fits, .png, .jpg) and catalog files (.cat)
        out_dir (str): Directory to place the genreated web page and associated
                       subdirectories
        zoom (int): The maximum zoom to tile images to. This generally doesn't
                    need to be set unless you have very very large images
        title (str): The title to placed on the webpage
        task_procs (int): The number of tasks to run in parallel
        procs_per_task (int): The number of tiles to process in parallel
        cat_wcs_fits_file (str): A fits file that has the WCS that will be used
                                 to map ra and dec coordinates from the catalog
                                 files to x and y coordinates in the map
        tile_size (Tuple[int, int]): The tile size for the leaflet map. Currently
                                     only [256, 256] is supported.
        image_engine (str): The method to convert array segments into png images
                            the IMG_ENGINE_PIL uses PIL and is faster,
                            but requires that the array be scaled before hand.
                            IMG_ENGINE_MPL uses matplotlib and is slower but can
                            scales the tiles according to the min/max of the
                            overall array.
    Returns:
        None
    """

    if len(files) == 0:
        raise ValueError("No files provided `files` is an empty list")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img_f_kwargs = dict(
        tile_size=tile_size,
        zoom=zoom,
        image_engine=image_engine,
        out_dir=out_dir,
        mp_procs=procs_per_task,
    )

    img_files = filter_on_extension(files, IMG_FORMATS)
    map_layer_names = list(map(get_map_layer_name, img_files))
    img_job_f = partial(tile_img, **img_f_kwargs)

    cat_files = filter_on_extension(files, CAT_FORMAT)
    marker_file_names = list(map(get_marker_file_names, cat_files))

    if len(cat_files) > 0:
        if cat_wcs_fits_file is None:
            cat_files = marker_file_names = []
            err_msg = [
                "Catalog files (.cat) were included, but no value was given for"
                "`cat_wcs_fits_file`. ra/dec can't be converted without a fits"
                "file. Skipping catalog conversion"
            ]
            print(" ".join(err_msg))
        else:
            cat_job_f = partial(catalog_to_markers, cat_wcs_fits_file, out_dir)
    else:
        cat_job_f = None

    pbar_locations = count(0)

    img_tasks = zip(repeat(img_job_f), zip(img_files, pbar_locations))
    cat_tasks = zip(repeat(cat_job_f), zip(cat_files, pbar_locations))
    tasks = chain(img_tasks, cat_tasks)

    if task_procs:
        q = JoinableQueue()
        any(map(lambda t: q.put(t), tasks))

        workers = [Process(target=async_worker, args=[q]) for _ in range(task_procs)]
        [w.start() for w in workers]  # can use any-map if this returns None

        q.join()
    else:
        any(map(lambda func_args: func_args[0](*func_args[1]), tasks))

    ns = "\n" * next(pbar_locations)
    print(ns + "Building index.html")
    web_map = Map(out_dir, title)
    any(map(web_map.add_tile_layer, map_layer_names))
    any(map(web_map.add_marker_catalog, marker_file_names))
    web_map.build_map()
    print("Done")


def dir_to_map(
    directory: str,
    out_dir: str = ".",
    exclude_predicate: Callable = lambda f: False,
    zoom: int = None,
    title: str = "FitsMap",
    task_procs: int = 0,
    procs_per_task: int = 0,
    cat_wcs_fits_file: str = None,
    tile_size: Shape = [256, 256],
    image_engine: str = IMG_ENGINE_PIL,
):
    """Converts a list of files into a LeafletJS map.

    Args:
        directory (str): Path to directory containing the files to be converted
        out_dir (str): Directory to place the genreated web page and associated
                       subdirectories
        exclude_predicate (Callable): A function that is applied to every file
                                      in ``directory`` and returns True if the
                                      file should not be processed as a part of
                                      the map, and False if it should be
                                      processed
        zoom (int): The maximum zoom to tile images to. This generally doesn't
                    need to be set unless you have very very large images
        title (str): The title to placed on the webpage
        task_procs (int): The number of tasks to run in parallel
        procs_per_task (int): The number of tiles to process in parallel
        cat_wcs_fits_file (str): A fits file that has the WCS that will be used
                                 to map ra and dec coordinates from the catalog
                                 files to x and y coordinates in the map. Note,
                                 that this file isn't subject to the
                                 ``exlclude_predicate``, so you can exclude a
                                 fits file from being tiled, but still use its
                                 header for WCS.
        tile_size (Tuple[int, int]): The tile size for the leaflet map. Currently
                                     only [256, 256] is supported.
        image_engine (str): The method to convert array segments into png images
                            the IMG_ENGINE_PIL uses PIL and is faster,
                            but requires that the array be scaled before hand.
                            IMG_ENGINE_MPL uses matplotlib and is slower but can
                            scales the tiles according to the min/max of the
                            overall array.
    Returns:
        None

    Raises:
        ValueError if the dir is empty, there are no convertable files or if
        ``exclude_predicate`` exlcudes all files
    """

    dir_files = list(
        map(
            lambda d: os.path.join(directory, d),
            filterfalse(exclude_predicate, os.listdir(directory),),
        )
    )

    if len(dir_files) == 0:
        raise ValueError(
            "No files in `directory` or `exlcude_predicate exlucdes everything"
        )

    files_to_map(
        dir_files,
        out_dir=out_dir,
        zoom=zoom,
        title=title,
        task_procs=task_procs,
        procs_per_task=procs_per_task,
        cat_wcs_fits_file=cat_wcs_fits_file,
        tile_size=tile_size,
        image_engine=image_engine,
    )
