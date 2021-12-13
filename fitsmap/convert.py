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

"""Converts image files and catalogs into a leafletJS map."""

import copy
import csv
import json
import multiprocessing as mp
import os
import shutil
import sys
from functools import partial
from itertools import chain, count, filterfalse, islice, product, repeat
from multiprocessing import JoinableQueue, Pool, Process
from queue import Empty, Queue
from typing import Any, Callable, Dict, Iterable, List, Tuple

import mapbox_vector_tile as mvt
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sharedmem
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from imageio import imread
from PIL import Image
from fitsmap.supercluster import Supercluster
from tqdm import tqdm

import fitsmap.utils as utils
import fitsmap.cartographer as cartographer

# https://github.com/zimeon/iiif/issues/11#issuecomment-131129062
Image.MAX_IMAGE_PIXELS = sys.maxsize

Shape = Tuple[int, int]

IMG_FORMATS = ["fits", "jpg", "png"]
CAT_FORMAT = ["cat"]
IMG_ENGINE_PIL = "PIL"
IMG_ENGINE_MPL = "MPL"
MPL_CMAP = "gray"

# MPL SINGLETON ENGINE =========================================================
mpl_f, mpl_img, mpl_alpha_f, mpl_norm = None, None, None, None
# ==============================================================================

MIXED_WHITESPACE_DELIMITER = "mixed_ws"
LOAD_CATALOG_BEFORE_PARSING = False


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
    shape: Tuple[int, int], zoom: int, tile_size: int
) -> Iterable[Tuple[int, int, int, slice, slice]]:

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


def balance_array(array: np.ndarray) -> np.ndarray:
    """Pads input array with zeros so that the long side is a multiple of the short side.

    Args:
        array (np.ndarray): array to balance
    Returns:
        a balanced version of ``array``
    """
    dim0, dim1 = array.shape[0], array.shape[1]

    exp_val = np.ceil(np.log2(max(dim0, dim1)))
    total_size = 2 ** exp_val
    pad_dim0 = int(total_size - dim0)
    pad_dim1 = int(total_size - dim1)

    if pad_dim0 > 0 or pad_dim1 > 0:
        if len(array.shape) == 3:
            padding = [[0, pad_dim0], [0, pad_dim1], [0, 0]]
        else:
            padding = [[0, pad_dim0], [0, pad_dim1]]

        return np.pad(
            array.astype(np.float32), padding, mode="constant", constant_values=np.nan
        )
    else:
        return array


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
        if len(shape) != 2:
            raise ValueError("FitsMap only supports 2D FITS files.")
    else:
        array = np.flipud(imread(file_location))
        # array = imread(file_location)

        if len(array.shape) == 3:
            shape = array.shape[:-1]
        elif len(array.shape) == 2:
            shape = array.shape
        else:
            # TODO: not sure this is ever reachable
            raise ValueError("FitsMap only supports 2D and 3D images.")

    return balance_array(array)


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


def make_dirs(out_dir: str, min_zoom: int, max_zoom: int) -> None:
    """Builds the directory tree for storing image tiles.

    Args:
        out_dir (str): The root directory to generate the tree in
        min_zoom (int): The minimum zoom level the image will be tiled at
        max_zoom (int): The maximum zoom level the image will be tiled at
    Returns:
        None
    """

    row_count = lambda z: int(np.sqrt(4 ** z))

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


def get_total_tiles(min_zoom: int, max_zoom: int) -> int:
    """Returns the total number of tiles that will be generated from an image.

    Args:
        min_zoom (int): The minimum zoom level te image will be tiled at
        max_zoom (int): The maximum zoom level te image will be tiled at
    Returns:
        The total number of tiles that will be generated
    """

    return int(sum([4 ** i for i in range(min_zoom, max_zoom + 1)]))


def make_tile_mpl(
    out_dir: str, array: np.ndarray, job: Tuple[int, int, int, slice, slice],
) -> None:
    """Extracts a tile from ``array`` and saves it at the proper place in ``out_dir`` using Matplotlib.

    Args:
        out_dir (str): The directory to save tile in
        array (np.ndarray): Array to extract a slice from
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

    tile = array[slice_ys, slice_xs].copy()

    if np.all(np.isnan(tile)):
        return

    global mpl_f
    global mpl_img
    global mpl_alpha_f
    global mpl_norm

    if mpl_f:
        # this is a singleton and starts out as null
        mpl_img.set_data(mpl_alpha_f(tile))  # pylint: disable=not-callable
        mpl_f.savefig(
            img_path, dpi=256, bbox_inches=0, facecolor=(0, 0, 0, 0),
        )
    else:
        if len(array.shape) == 2:
            cmap = copy.copy(mpl.cm.get_cmap(MPL_CMAP))
            cmap.set_bad(color=(0, 0, 0, 0))

            img_kwargs = dict(
                origin="lower", cmap=cmap, interpolation="nearest", norm=mpl_norm
            )

            mpl_alpha_f = lambda arr: arr
        else:
            img_kwargs = dict(interpolation="nearest", origin="lower", norm=mpl_norm)

            def adjust_pixels(arr):
                img = arr.copy()
                if img.shape[2] == 3:
                    img = np.concatenate(
                        (
                            img,
                            np.ones(list(img.shape[:-1]) + [1], dtype=np.float32) * 255,
                        ),
                        axis=2,
                    )

                ys, xs = np.where(np.isnan(img[:, :, 0]))
                img[ys, xs, :] = np.array([0, 0, 0, 0], dtype=np.float32)

                return img.astype(np.uint8)

            mpl_alpha_f = lambda arr: adjust_pixels(arr)

        mpl_f = plt.figure(dpi=256)
        mpl_f.set_size_inches([256 / 256, 256 / 256])
        mpl_img = plt.imshow(mpl_alpha_f(tile), **img_kwargs)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis("off")
        mpl_f.savefig(
            img_path, dpi=256, bbox_inches=0, facecolor=(0, 0, 0, 0),
        )


def make_tile_pil(
    out_dir: str, array: np.ndarray, job: Tuple[int, int, int, slice, slice]
) -> None:
    """Extracts a tile from ``array`` and saves it at the proper place in ``out_dir`` using PIL.

    Args:
        out_dir (str): The directory to save tile in
        array (np.ndarray): Array to extract a slice from
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

    tile = array[slice_ys, slice_xs].copy()

    if np.all(np.isnan(tile)):
        return

    if len(tile.shape) < 3:
        tile = np.dstack([tile, tile, tile, np.ones_like(tile) * 255])
    elif tile.shape[2] == 3:
        tile = np.concatenate(
            (tile, np.ones(list(tile.shape[:-1]) + [1], dtype=np.float32) * 255),
            axis=2,
        )

    ys, xs = np.where(np.isnan(tile[:, :, 0]))
    tile[ys, xs, :] = np.array([0, 0, 0, 0], dtype=np.float32)
    img = Image.fromarray(np.flipud(tile).astype(np.uint8))
    del tile

    img.thumbnail([256, 256], Image.LANCZOS)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    img.save(img_path, "PNG")
    del img


def tile_img(
    file_location: str,
    pbar_loc: int,
    tile_size: Shape = [256, 256],
    min_zoom: int = 0,
    image_engine: str = IMG_ENGINE_MPL,
    out_dir: str = ".",
    mp_procs: int = 0,
    norm_kwargs: dict = {},
) -> None:
    """Extracts tiles from the array at ``file_location``.

    Args:
        file_location (str): The file location of the image to tile
        pbar_loc (int): The index of the location of to print the tqdm bar
        tile_size (Tuple[int, int]): The pixel size of the tiles in the map
        min_zoom (int): The minimum zoom to create tiles for. The default value
                        is 0, but if it can be helpful to set it to a value
                        greater than zero if your running out of memory as the
                        lowest zoom images can be the most memory intensive.
        img_engine (str): Method to convert array tile to an image. Can be one
                          of convert.IMAGE_ENGINE_PIL (using pillow(PIL)) or
                          of convert.IMAGE_ENGINE_MPL (using matplotlib)
        out_dir (str): The root directory to save the tiles in
        mp_procs (int): The number of multiprocessing processes to use for
                        generating tiles.
        norm_kwargs (dict): Optional normalization keyword arguments passed to
                            `astropy.visualization.simple_norm`. The default is
                            linear scaling using min/max values. See documentation
                            for more information: https://docs.astropy.org/en/stable/api/astropy.visualization.mpl_normalize.simple_norm.html

    Returns:
        None
    """

    _, fname = os.path.split(file_location)
    if get_map_layer_name(file_location) in os.listdir(out_dir):
        print(f"{fname} already tiled. Skipping tiling.")
        return

    # reset mpl vars just in case they have been set by another img
    global mpl_f
    global mpl_img
    global mpl_norm

    if mpl_f:
        mpl_f = None
        mpl_img = None

    # get image
    array = get_array(file_location)
    mpl_norm = simple_norm(array, **norm_kwargs)

    zooms = get_zoom_range(array.shape, tile_size)
    min_zoom = max(min_zoom, zooms[0])
    max_zoom = zooms[1]

    # build directory structure
    name = get_map_layer_name(file_location)
    tile_dir = os.path.join(out_dir, name)
    if name not in os.listdir(out_dir):
        os.mkdir(tile_dir)

    make_dirs(tile_dir, min_zoom, max_zoom)

    tile_params = chain.from_iterable(
        [
            slice_idx_generator(array.shape, z, 256 * (2 ** i))
            for (i, z) in enumerate(range(max_zoom, min_zoom - 1, -1), start=0)
        ]
    )

    # tile the image
    total_tiles = get_total_tiles(min_zoom, max_zoom)

    if image_engine == IMG_ENGINE_MPL:
        make_tile = partial(make_tile_mpl, tile_dir)
    else:
        make_tile = partial(make_tile_pil, tile_dir)

    if mp_procs:
        mp_array = sharedmem.empty_like(array)
        mp_array[:] = array[:]
        work = partial(make_tile, mp_array)
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
                        disable=bool(os.getenv("DISBALE_TQDM", False)),
                    ),
                )
            )
    else:
        work = partial(make_tile, array)
        any(
            map(
                work,
                tqdm(
                    tile_params,
                    desc="Converting " + name,
                    total=total_tiles,
                    unit="tile",
                    position=pbar_loc,
                    disable=bool(os.getenv("DISBALE_TQDM", False)),
                ),
            )
        )

    if image_engine == IMG_ENGINE_MPL:
        plt.close("all")


def get_map_layer_name(file_location: str) -> str:
    """Tranforms a ``file_location`` into the javascript layer name.

    Args:
        file_location (str): The file location to convert

    Returns:
        The javascript name that will be used in the HTML map
    """
    _, fname = os.path.split(file_location)
    name = (
        os.path.splitext(fname)[0]
        .replace(".", "_")
        .replace("-", "_")
        .replace("(", "_")
        .replace(")", "_")
    )
    return name


def get_marker_file_name(file_location: str) -> str:
    """Tranforms a ``file_location`` into the javascript marker file name.

    Args:
        file_location (str): The file location to convert

    Returns:
        The javascript name that will be used in the HTML map
    """
    return os.path.split(file_location)[1] + ".js"


def line_to_cols(raw_col_vals: str) -> List[str]:
    """Transform a raw text line of column names into a list of column names

    Args:
        raw_line (str): String from textfile

    Returns:
        A list of the column names in order
    """
    change_case = [
        "RA",
        "DEC",
        "Ra",
        "Dec",
        "X",
        "Y",
        "ID",
        "iD",
        "Id",
        "A",
        "B",
        "THETA",
        "Theta",
    ]

    # make ra and dec lowercase for ease of access
    raw_cols = list(map(lambda s: s.lower() if s in change_case else s, raw_col_vals,))

    # if header line starts with a '#' exclude it
    if raw_cols[0] == "#":
        return raw_cols[1:]
    else:
        return raw_cols


def line_to_json(
    wcs: WCS, columns: List[str], catalog_assets_path: str, src_vals: List[str],
) -> Dict[str, Any]:
    """Transform a raw text line attribute values into a JSON marker

    Args:
        raw_line (str): String from the marker file

    Returns:
        A list of the column names in order
    """
    src_id = str(src_vals[columns.index("id")])
    if "x" in columns and "y" in columns:
        img_x = float(src_vals[columns.index("x")])
        img_y = float(src_vals[columns.index("y")])
    else:
        ra = float(src_vals[columns.index("ra")])
        dec = float(src_vals[columns.index("dec")])

        # We assume the origins of the images for catalog conversion start at (1,1).
        [[img_x, img_y]] = wcs.wcs_world2pix([[ra, dec]], 1)

    if "a" in columns and "b" in columns and "theta" in columns:
        a = float(src_vals[columns.index("a")])
        b = float(src_vals[columns.index("b")])
        if np.isnan(b):
            b = a

        theta = float(src_vals[columns.index("theta")])
        if np.isnan(theta):
            theta = 0

    else:
        a = -1
        b = -1
        theta = -1

    # The default catalog convention is that the lower left corner is (1, 1)
    # The default leaflet convention is that the lower left corner is (0, 0)
    # The convention for leaflet is to place markers at the lower left corner
    # of a pixel, to center the marker on the pixel, we subtract 1 to bring it
    # to leaflet convention and add 0.5 to move it to the center of the pixel.
    x = (img_x - 1) + 0.5
    y = (img_y - 1) + 0.5

    src_desc = {k: v for k, v in zip(columns, src_vals)}

    src_desc["fm_y"] = y
    src_desc["fm_x"] = x
    src_desc["fm_cat"] = catalog_assets_path.split(os.sep)[-1]

    src_json = os.path.join(catalog_assets_path, f"{src_id}.json")
    with open(src_json, "w") as f:
        json.dump(src_desc, f, separators=(",", ":"))  # no whitespace

    return dict(
        geometry=dict(coordinates=[x, y],),
        tags=dict(
            a=a,
            b=b,
            theta=theta,
            catalog_id=src_id,
            cat_path=os.path.basename(catalog_assets_path),
        ),
    )


def process_catalog_file_chunk(
    process_f: Callable, fname: str, delimiter: str, start: int, end: int
) -> List[dict]:
    # newline="" for csv reader, see
    # https://docs.python.org/3/library/csv.html#csv.reader
    f = open(fname, "r", newline="")
    f.seek(start)
    f.readline()  # id start==0 skip cols, else advance to next complete line

    json_lines = []

    if LOAD_CATALOG_BEFORE_PARSING:
        raw_lines = f.readlines(end - start)
        reader = csv.reader(raw_lines, delimiter=delimiter, skipinitialspace=True)

        update_every = 1000
        count = 1
        for line in reader:
            json_lines.append(process_f(line))
            count += 1
            if count % update_every == 0:
                process_catalog_file_chunk.q.put(count)
                count = 1

    else:
        raw_lines = []
        reader = csv.reader(raw_lines, delimiter=delimiter, skipinitialspace=True)
        current = f.tell()
        update_every = 10000
        count = 1
        while current < end:
            raw_lines.append(f.readline().strip())
            processed_lines = next(reader)
            json_lines.append(process_f(processed_lines))
            current = f.tell()
            count += 1
            if count % update_every == 0:
                process_catalog_file_chunk.q.put(count)
                count = 1

        if count > 1:
            process_catalog_file_chunk.q.put(count)
    f.close()
    return json_lines


# This allows the queue reference to be shared between processes
# https://stackoverflow.com/a/3843313/2691018
def process_catalog_file_chunk_init(q):
    process_catalog_file_chunk.q = q


def _simplify_mixed_ws(catalog_fname: str) -> None:
    with open(catalog_fname, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    with open(catalog_fname, "w") as f:
        for line in lines:
            f.write(" ".join([token.strip() for token in line.split()]) + "\n")


def procbar_listener(q: Queue, bar: tqdm) -> None:
    while True:
        update = q.get()
        if update:
            bar.update(n=update)
        else:
            break


def make_marker_tile(
    cluster: Supercluster, out_dir: str, zyx: Tuple[int, Tuple[int, int]],
) -> None:
    z, (y, x) = zyx

    if not os.path.exists(os.path.join(out_dir, str(z))):
        os.mkdir(os.path.join(out_dir, str(z)))

    if not os.path.exists(os.path.join(out_dir, str(z), str(y))):
        os.mkdir(os.path.join(out_dir, str(z), str(y)))

    out_path = os.path.join(out_dir, str(z), str(y), f"{x}.pbf")

    tile_sources = cluster.get_tile(z, x, y)

    if tile_sources:
        tile_sources["name"] = "Points"

        for i in range(len(tile_sources["features"])):
            # tile_sources["features"][i]["geometry"] = "POINT({} {})".format(
            #     *list(map(
            #         int,
            #         tile_sources["features"][i]["geometry"]
            #     ))
            # )
            tile_sources["features"][i]["geometry"] = "POINT(0 0)"  # we dont' use this

        # with open(out_path.replace("pbf", "json"), "w") as f:
        #     json.dump(tile_sources, f, indent=2)

        encoded_tile = mvt.encode([tile_sources], extents=256)

        with open(out_path, "wb") as f:
            f.write(encoded_tile)


def tile_markers(
    wcs_file: str,
    out_dir: str,
    catalog_delim: str,
    mp_procs: int,
    prefer_xy: bool,
    min_zoom: int,
    max_zoom: int,
    tile_size: int,
    max_x: int,
    max_y: int,
    catalog_file: str,
    pbar_loc: int,
) -> None:
    _, fname = os.path.split(catalog_file)

    catalog_layer_name = get_map_layer_name(catalog_file)
    if catalog_layer_name in os.listdir(out_dir):
        print(f"{fname} already tiled. Skipping tiling.")
        return
    else:
        os.mkdir(os.path.join(out_dir, catalog_layer_name))

    if catalog_delim == MIXED_WHITESPACE_DELIMITER:
        _simplify_mixed_ws(catalog_file)
        catalog_delim = " "

    with open(catalog_file, "r", newline="") as f:
        csv_reader = csv.reader(f, delimiter=catalog_delim, skipinitialspace=True)
        columns = line_to_cols(next(csv_reader))

    ra_dec_coords = "ra" in columns and "dec" in columns
    x_y_coords = "x" in columns and "y" in columns
    use_xy = (not ra_dec_coords) or (prefer_xy)

    if (not ra_dec_coords and not x_y_coords) or "id" not in columns:
        err_msg = " ".join(
            [
                catalog_file + " is missing coordinate columns (ra/dec, xy),",
                "an 'id' column, or all of the above",
            ]
        )
        raise ValueError(err_msg)

    if (not use_xy) and (wcs_file is None):
        err_msg = " ".join(
            [catalog_file + " uses ra/dec coords, but a WCS file wasn't", "provided."]
        )
        raise ValueError(err_msg)

    wcs = WCS(wcs_file) if wcs_file else None

    catalog_assets_parent_path = os.path.join(out_dir, "catalog_assets")
    if "catalog_assets" not in os.listdir(out_dir):
        os.mkdir(catalog_assets_parent_path)

    catalog_assets_path = os.path.join(catalog_assets_parent_path, catalog_layer_name)
    if catalog_layer_name not in os.listdir(catalog_assets_parent_path):
        os.mkdir(catalog_assets_path)

    line_func = partial(line_to_json, wcs, columns, catalog_assets_path,)

    bar = tqdm(
        position=pbar_loc,
        desc="Parsing " + catalog_file,
        disable=bool(os.getenv("DISBALE_TQDM", False)),
    )

    # this queue manages a proc bar instance that can be shared among multiple
    # processes, i have a feeling there is a better way to do this
    q = mp.Queue()
    monitor = mp.Process(target=procbar_listener, args=(q, bar))
    monitor.start()

    process_f = partial(
        process_catalog_file_chunk, line_func, catalog_file, catalog_delim,
    )

    catalog_file_size = os.path.getsize(catalog_file)

    if mp_procs > 1:
        # split the file by splitting the byte size of the file into even sized
        # chunks. We always read to the end of the first line so its ok to get
        # dropped into the middle of a link
        boundaries = np.linspace(0, catalog_file_size, mp_procs + 1, dtype=np.int64)
        file_chunk_pairs = list(zip(boundaries[:-1], boundaries[1:]))

        # To keep the progress bar up to date we need an extra process that
        # montiors the queue for updates from the workers. The extra process
        # doesn't do much and so it shouldn't be a strain to add it.

        with Pool(mp_procs, process_catalog_file_chunk_init, [q]) as p:
            catalog_values = list(
                chain.from_iterable(p.starmap(process_f, file_chunk_pairs))
            )
    else:
        process_catalog_file_chunk_init(q)
        catalog_values = process_f(0, catalog_file_size)

    # this kills the iter in the monitor process
    q.put(None)
    monitor.join()
    del monitor

    # TEMP FIX FOR DREAM
    # max_zoom += 2
    bar = tqdm(
        position=pbar_loc,
        desc="Clustering " + catalog_file + "(THIS MAY TAKE A WHILE)",
        disable=bool(os.getenv("DISBALE_TQDM", False)),
        total=max_zoom + 1 - min_zoom,
    )

    q = mp.Queue()
    monitor = mp.Process(target=procbar_listener, args=(q, bar))
    monitor.start()

    # cluster the parsed sources
    # need to get super cluster stuff in here
    clusterer = Supercluster(
        min_zoom=min_zoom,
        max_zoom=max_zoom - 1,
        extent=tile_size,
        radius=max(max(max_x, max_y) / tile_size, 40),
        node_size=np.log2(len(catalog_values)) * 2,
        alternate_CRS=(max_x, max_y),
        update_f=lambda: q.put(1),
        log=True,
    ).load(catalog_values)

    # this kills the iter in the monitor process
    q.put(None)
    monitor.join()
    del monitor

    # tile the sources and save using protobuf
    zs = range(min_zoom, max_zoom + 1)
    ys = [range(2 ** z) for z in zs]
    xs = [range(2 ** z) for z in zs]
    tile_idxs = list(
        chain.from_iterable(
            [zip(repeat(zs[i]), product(ys[i], xs[i])) for i in range(len(zs))]
        )
    )

    clusterer.update_f = None
    clusterer.log = False

    tile_f = partial(
        make_marker_tile, clusterer, os.path.join(out_dir, catalog_layer_name),
    )

    tracked_collection = tqdm(
        tile_idxs,
        position=pbar_loc,
        desc="Tiling " + catalog_file,
        disable=bool(os.getenv("DISBALE_TQDM", False)),
    )

    # the super cluster object can be very large, so this may not be worth
    # eating up memory, set to False
    if False:
        with Pool(mp_procs) as p:
            list(p.imap_unordered(tile_f, tracked_collection, chunksize=100))
    else:
        list(map(tile_f, tracked_collection))


def async_worker(q: JoinableQueue) -> None:
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
    min_zoom: int = 0,
    title: str = "FitsMap",
    task_procs: int = 0,
    procs_per_task: int = 0,
    catalog_delim: str = ",",
    cat_wcs_fits_file: str = None,
    max_catalog_zoom: int = -1,
    tile_size: Tuple[int, int] = [256, 256],
    image_engine: str = IMG_ENGINE_MPL,
    norm_kwargs: dict = {},
    rows_per_column: int = np.inf,
    prefer_xy: bool = False,
) -> None:
    """Converts a list of files into a LeafletJS map.

    Args:
        files (List[str]): List of files to convert into a map, can include image
                           files (.fits, .png, .jpg) and catalog files (.cat)
        out_dir (str): Directory to place the genreated web page and associated
                       subdirectories
        min_zoom (int): The minimum zoom to create tiles for. The default value
                        is 0, but if it can be helpful to set it to a value
                        greater than zero if your running out of memory as the
                        lowest zoom images can be the most memory intensive.
        title (str): The title to placed on the webpage
        task_procs (int): The number of tasks to run in parallel
        procs_per_task (int): The number of tiles to process in parallel
        catalog_delim (str): The delimited for catalog (.cat) files. Deault is
                             whitespace.
        cat_wcs_fits_file (str): A fits file that has the WCS that will be used
                                 to map ra and dec coordinates from the catalog
                                 files to x and y coordinates in the map
        max_catalog_zoom (int): The zoom level to stop clustering on, the
                                default is the max zoom level of the image. For
                                images with a high source density, setting this
                                higher than the max zoom will help with
                                performance.
        tile_size (Tuple[int, int]): The tile size for the leaflet map. Currently
                                     only [256, 256] is supported.
        image_engine (str): The method that converts the image data into image
                            tiles. The default is convert.IMG_ENGINE_MPL
                            (matplotlib) the other option is
                            convert.IMG_ENGINE_PIL (pillow). Pillow can render
                            FITS files but doesn't do any scaling. Pillow may
                            be more performant for only PNG images.
        norm_kwargs (dict): Optional normalization keyword arguments passed to
                            `astropy.visualization.simple_norm`. The default is
                            linear scaling using min/max values. See documentation
                            for more information: https://docs.astropy.org/en/stable/api/astropy.visualization.mpl_normalize.simple_norm.html
        rows_per_column (int): If converting a catalog, the number of items in
                               have in each column of the marker popup.
                               By default produces all values in a single
                               column. Setting this value can make it easier to
                               work with catalogs that have a lot of values for
                               each object.
    Returns:
        None
    """

    if len(files) == 0:
        raise ValueError("No files provided `files` is an empty list")

    unlocatable_files = list(filterfalse(os.path.exists, files))
    if len(unlocatable_files) > 0:
        raise FileNotFoundError(unlocatable_files)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if "js" not in os.listdir(out_dir):
        os.mkdir(os.path.join(out_dir, "js"))

    if "css" not in os.listdir(out_dir):
        os.mkdir(os.path.join(out_dir, "css"))

    img_f_kwargs = dict(
        tile_size=tile_size,
        min_zoom=min_zoom,
        image_engine=image_engine,
        out_dir=out_dir,
        mp_procs=procs_per_task,
        norm_kwargs=norm_kwargs,
    )

    img_files = filter_on_extension(files, IMG_FORMATS)
    img_layer_names = list(map(get_map_layer_name, img_files))
    img_job_f = partial(tile_img, **img_f_kwargs)

    cat_files = filter_on_extension(files, CAT_FORMAT)
    cat_layer_names = list(map(get_map_layer_name, cat_files))

    max_x, max_y = utils.peek_image_info(img_files)
    max_dim = max(max_x, max_y)
    if len(cat_files) > 0:
        # get highlevel image info for catalogging function
        max_zoom = int(np.log2(2 ** np.ceil(np.log2(max_dim)) / tile_size[0]))
        max_dim = 2 ** max_zoom * tile_size[0]
        if max_catalog_zoom == -1:
            max_zoom = int(np.log2(2 ** np.ceil(np.log2(max_dim)) / tile_size[0]))
        else:
            max_zoom = max_catalog_zoom

        cat_job_f = partial(
            tile_markers,
            cat_wcs_fits_file,
            out_dir,
            catalog_delim,
            procs_per_task,
            prefer_xy,
            min_zoom,
            max_zoom,
            tile_size[0],
            max_dim,
            max_dim,
        )
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

    ns = "\n" * (next(pbar_locations) - 1)
    print(ns + "Building index.html")

    if cat_wcs_fits_file is not None:
        cat_wcs = WCS(cat_wcs_fits_file)
    else:
        cat_wcs = None

    cartographer.chart(
        out_dir,
        title,
        img_layer_names,
        cat_layer_names,
        cat_wcs,
        rows_per_column,
        (max_x, max_y),
    )
    print("Done.")


def dir_to_map(
    directory: str,
    out_dir: str = ".",
    exclude_predicate: Callable = lambda f: False,
    min_zoom: int = 0,
    title: str = "FitsMap",
    task_procs: int = 0,
    procs_per_task: int = 0,
    catalog_delim: str = ",",
    cat_wcs_fits_file: str = None,
    max_catalog_zoom: int = -1,
    tile_size: Shape = [256, 256],
    image_engine: str = IMG_ENGINE_MPL,
    norm_kwargs: dict = {},
    rows_per_column: int = np.inf,
) -> None:
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
        min_zoom (int): The minimum zoom to create tiles for. The default value
                        is 0, but if it can be helpful to set it to a value
                        greater than zero if your running out of memory as the
                        lowest zoom images can be the most memory intensive.
        title (str): The title to placed on the webpage
        task_procs (int): The number of tasks to run in parallel
        procs_per_task (int): The number of tiles to process in parallel
        catalog_delim (str): The delimiter for catalog (.cat) files. Deault is
                             comma.
        cat_wcs_fits_file (str): A fits file that has the WCS that will be used
                                 to map ra and dec coordinates from the catalog
                                 files to x and y coordinates in the map. Note,
                                 that this file isn't subject to the
                                 ``exlclude_predicate``, so you can exclude a
                                 fits file from being tiled, but still use its
                                 header for WCS.
        max_catalog_zoom (int): The zoom level to stop clustering on, the
                                default is the max zoom level of the image. For
                                images with a high source density, setting this
                                higher than the max zoom will help with
                                performance.
        tile_size (Tuple[int, int]): The tile size for the leaflet map. Currently
                                     only [256, 256] is supported.
        image_engine (str): The method that converts the image data into image
                            tiles. The default is convert.IMG_ENGINE_MPL
                            (matplotlib) the other option is
                            convert.IMG_ENGINE_PIL (pillow). Pillow can render
                            FITS files but doesn't do any scaling. Pillow may
                            be more performant for only PNG images.
        norm_kwargs (dict): Optional normalization keyword arguments passed to
                            `astropy.visualization.simple_norm`. The default is
                            linear scaling using min/max values. See documentation
                            for more information: https://docs.astropy.org/en/stable/api/astropy.visualization.mpl_normalize.simple_norm.html
        rows_per_column (int): If converting a catalog, the number of items in
                               have in each column of the marker popup.
                               By default produces all values in a single
                               column. Setting this value can make it easier to
                               work with catalogs that have a lot of values for
                               each object.
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
            "No files in `directory` or `exclude_predicate` excludes everything"
        )

    files_to_map(
        sorted(dir_files),
        out_dir=out_dir,
        min_zoom=min_zoom,
        title=title,
        task_procs=task_procs,
        procs_per_task=procs_per_task,
        catalog_delim=catalog_delim,
        cat_wcs_fits_file=cat_wcs_fits_file,
        max_catalog_zoom=max_catalog_zoom,
        tile_size=tile_size,
        image_engine=image_engine,
        norm_kwargs=norm_kwargs,
        rows_per_column=rows_per_column,
    )
