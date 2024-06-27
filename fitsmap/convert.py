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

"""Converts image files and catalogs into a leafletJS map."""

import copy
import csv
import io
import logging
import os
import sys
from functools import partial
from itertools import (
    chain,
    filterfalse,
    groupby,
    islice,
    product,
    repeat,
    starmap,
)
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import cbor2
import mapbox_vector_tile as mvt
import matplotlib as mpl

mpl.use("Agg")  # need to use this for processes safe matplotlib
import astropy
import matplotlib.pyplot as plt
import numpy as np
import ray
import ray.util.queue as queue
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from PIL import Image

import fitsmap.utils as utils
import fitsmap.cartographer as cartographer
import fitsmap.padded_array as pa
from fitsmap.output_manager import OutputManager
from fitsmap.supercluster import Supercluster

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


def balance_array(array: np.ndarray) -> pa.PaddedArray:
    """Pads input array with zeros so that the long side is a multiple of the short side.

    Args:
        array (np.ndarray): array to balance
    Returns:
        a balanced version of ``array`` via a PaddedArray
    """
    dim0, dim1 = array.shape[0], array.shape[1]

    exp_val = np.ceil(np.log2(max(dim0, dim1)))
    total_size = 2**exp_val
    pad_dim0 = int(total_size - dim0)
    pad_dim1 = int(total_size - dim1)

    return pa.PaddedArray(array, (pad_dim0, pad_dim1))


def get_array(file_location: str) -> pa.PaddedArray:
    """Opens the array at ``file_location`` can be an image or a fits file

    Args:
        file_location (str): the path to the image
    Returns:
        A numpy array representing the image.
    """

    _, ext = os.path.splitext(file_location)

    if ext == ".fits":
        array = fits.getdata(file_location).astype(np.float32)
        shape = array.shape
        if len(shape) != 2:
            raise ValueError("FitsMap only supports 2D FITS files.")
    else:
        array = np.flipud(Image.open(file_location)).astype(np.float32)

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

    row_count = lambda z: int(np.sqrt(4**z))

    def build_z_ys(z, ys):
        list(
            map(
                lambda y: os.makedirs(os.path.join(out_dir, f"{z}/{y}"), exist_ok=True),
                ys,
            )
        )

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

    return int(sum([4**i for i in range(min_zoom, max_zoom + 1)]))


def imread_default(path: str, default: np.ndarray) -> np.ndarray:
    """Opens an image if it exists, if not returns a tranparent image.
    Args:
        path (str): Image file location
        size (int, optional): The image size, assumed square. Defaults to 256.
    Returns:
        np.ndarray: the image if it exists. if not, a transparent image of
                    size (size, size, 4).
    """
    try:
        return np.flipud(Image.open(path))
    except FileNotFoundError:
        return default


def make_tile_mpl(
    mpl_norm: mpl.colors.Normalize, mpl_cmap: mpl.colors.Colormap, tile: np.ndarray
) -> np.ndarray:
    """Converts array data into an image using matplotlib
    Args:
        mpl_f (mpl.figure.Figure): The matplotlib figure to use
        mpl_img (mpl.image.AxesImage): The matplotlib image to use
        mpl_alpha_f (Callable[[np.ndarray], np.ndarray]): A function that
                                                          converts the input
                                                          array into an RGBA
        tile (np.ndarray): The array data
    Returns:
        np.ndarray: The array data converted into an image using Matplotlib
    """
    if type(mpl_norm) == ray._raylet.ObjectRef:
        mpl_norm = ray.get(mpl_norm)
        mpl_cmap = ray.get(mpl_cmap)

    tile_image = mpl_cmap(mpl_norm(np.flipud(tile)))
    return Image.fromarray((tile_image * 255).astype(np.uint8))


def make_tile_pil(tile: np.ndarray) -> np.ndarray:
    """Converts the input array into an image using PIL
    Args:
        tile (np.ndarray): The array data to be converted
    Returns:
        np.ndarray: an RGBA version of the input data
    """

    if len(tile.shape) == 2:
        img_tile = np.dstack([tile, tile, tile, np.ones_like(tile) * 255])
    elif tile.shape[2] == 3:
        img_tile = np.concatenate(
            (tile, np.ones(list(tile.shape[:-1]) + [1], dtype=np.float32) * 255),
            axis=2,
        )
    else:
        img_tile = np.copy(tile)

    # else the image is already RGBA

    ys, xs = np.where(np.isnan(np.atleast_3d(tile)[:, :, 0]))
    img_tile[ys, xs, :] = np.array([0, 0, 0, 0], dtype=np.float32)
    img = Image.fromarray(np.flipud(img_tile).astype(np.uint8))

    return img


def mem_safe_make_tile(
    out_dir: str,
    tile_f: Callable[[np.ndarray], np.ndarray],
    array: np.ndarray,
    job: Union[
        Tuple[int, int, int, slice, slice], List[Tuple[int, int, int, slice, slice]]
    ],
) -> None:
    """Extracts a tile from ``array`` and saves it at the proper place in ``out_dir`` using PIL.
    Args:
        out_dir (str): The directory to save tile in
        tile_f (Callable[[np.ndarray], np.ndarray]): A function that converts a
                                                     subset of the image array
                                                     into an png tile. Is one
                                                     of `make_tile_pil` or
                                                     `make_tile_mpl`
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

    default_array = np.zeros([256, 256, 4], dtype=np.uint8)
    get_img = partial(imread_default, default=default_array)

    # If this is being run in parallel it will be a List of Tuples if its being
    # run serially it will be a single Tuple, wrap the Tuple in a list to keep
    # it simple
    jobs = [job] if isinstance(job, tuple) else job

    for z, y, x, slice_ys, slice_xs in jobs:
        try:
            img_path = build_path(z, y, x, out_dir)

            with_out_dir = partial(os.path.join, out_dir)

            if os.path.exists(with_out_dir(f"{z+1}")):
                top_left = get_img(with_out_dir(f"{z+1}", f"{2*y+1}", f"{2*x}.png"))
                top_right = get_img(with_out_dir(f"{z+1}", f"{2*y+1}", f"{2*x+1}.png"))
                bottom_left = get_img(with_out_dir(f"{z+1}", f"{2*y}", f"{2*x}.png"))
                bottom_right = get_img(with_out_dir(f"{z+1}", f"{2*y}", f"{2*x+1}.png"))

                tile = np.concatenate(
                    [
                        np.concatenate([top_left, top_right], axis=1),
                        np.concatenate([bottom_left, bottom_right], axis=1),
                    ],
                    axis=0,
                )
                img = Image.fromarray(np.flipud(tile))
            else:
                tile = array[slice_ys, slice_xs]
                img = tile_f(tile)

            # if the tile is all transparent, don't save to disk
            if np.any(np.array(img)[..., -1]):
                img.thumbnail([256, 256], Image.Resampling.LANCZOS)
                img.convert("RGBA").save(img_path, "PNG")
            else:
                pass
        except Exception as e:
            print("Tile creation failed for tile:", z, y, x, slice_ys, slice_xs)
            print(e)


def build_mpl_objects(
    array: np.ndarray, norm_kwargs: Dict[str, Any]
) -> Tuple[mpl.colors.Normalize, mpl.colors.Colormap]:
    """Builds the matplotlib objects that norm and convert fits data to RGB.

    Args:
        array (np.ndarray): The array to be tiled
        norm_kwargs (Dict[str, Any]): The kwargs to be passed to `simple_norm`
    Returns:
        Tuple[mpl.colors.Normalize, mpl.colors.Colormap]: The matplotlib objects
                                                         needed to create a tile
    """
    mpl_norm = simple_norm(array, **norm_kwargs)
    mpl_cmap = copy.copy(mpl.colormaps[MPL_CMAP])
    mpl_cmap.set_bad(color=(0, 0, 0, 0))
    return mpl_norm, mpl_cmap


def tile_img(
    file_location: str,
    pbar_ref: Tuple[int, queue.Queue],
    tile_size: Shape = [256, 256],
    min_zoom: int = 0,
    out_dir: str = ".",
    mp_procs: int = 0,
    norm_kwargs: dict = {},
    batch_size: int = 1000,
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
        out_dir (str): The root directory to save the tiles in
        mp_procs (int): The number of multiprocessing processes to use for
                        generating tiles.
        norm_kwargs (dict): Optional normalization keyword arguments passed to
                            `astropy.visualization.simple_norm`. The default is
                            linear scaling using min/max values. See documentation
                            for more information: https://docs.astropy.org/en/stable/api/astropy.visualization.mpl_normalize.simple_norm.html
        batch_size (int): The number of tiles to process at a time, when tiling
                          in parallel. The default is 1000.

    Returns:
        None
    """

    _, fname = os.path.split(file_location)
    if get_map_layer_name(file_location) in os.listdir(out_dir):
        OutputManager.write(pbar_ref, f"{fname} already tiled. Skipping tiling.")
        return

    # get image
    array = get_array(file_location)

    # if we're using matplotlib we need to instantiate the matplotlib objects
    # before we pass them to ray
    image_engine = IMG_ENGINE_MPL if file_location.endswith(".fits") else IMG_ENGINE_PIL
    if image_engine == IMG_ENGINE_MPL:
        image_norm = norm_kwargs.get(os.path.basename(file_location), norm_kwargs)
        mpl_norm, mpl_cmap = build_mpl_objects(array.array, image_norm)

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
            slice_idx_generator(array.shape, z, 256 * (2**i))
            for (i, z) in enumerate(range(max_zoom, min_zoom - 1, -1), start=0)
        ]
    )

    total_tiles = get_total_tiles(min_zoom, max_zoom)

    if mp_procs > 1:
        # We need to process batches to offset the cost of spinning up a process
        def batch_params(iter, batch_size):
            while True:
                batch = list(islice(iter, batch_size))
                if batch:
                    yield batch
                else:
                    break

        # Put the array in the ray object store. After it's there remove the
        # reference to it so that it can be garbage collected.
        arr_obj_id = ray.put(array)
        del array

        if image_engine == IMG_ENGINE_MPL:
            mpl_norm = ray.put(mpl_norm)
            mpl_cmap = ray.put(mpl_cmap)
            tile_f = partial(
                make_tile_mpl,
                mpl_norm,
                mpl_cmap,
            )
        else:
            tile_f = make_tile_pil

        make_tile_f = ray.remote(num_cpus=1)(mem_safe_make_tile)

        OutputManager.set_description(pbar_ref, f"Converting {name}")
        OutputManager.set_units_total(pbar_ref, unit="tile", total=total_tiles)

        # in parallel we have to go one zoom level at a time because the images
        # at lower zoom levels are dependent on the tiles are higher zoom levels
        for zoom, jobs in groupby(tile_params, lambda z: z[0]):
            if zoom == max_zoom:
                utils.backpressure_queue_ray(
                    make_tile_f.remote,
                    list(
                        zip(
                            repeat(tile_dir),
                            repeat(tile_f),
                            repeat(arr_obj_id),
                            batch_params(jobs, batch_size),
                        )
                    ),
                    pbar_ref,
                    mp_procs,
                    batch_size,
                )
            else:
                if "arr_obj_id" in locals():
                    del arr_obj_id
                work = list(
                    zip(
                        repeat(tile_dir),
                        repeat(make_tile_pil),
                        repeat(None),
                        batch_params(jobs, batch_size),
                    )
                )

                utils.backpressure_queue_ray(
                    make_tile_f.remote,
                    work,
                    pbar_ref,
                    mp_procs,
                    batch_size,
                )
    else:
        if image_engine == IMG_ENGINE_MPL:
            tile_f = partial(make_tile_mpl, mpl_norm, mpl_cmap)
        else:
            tile_f = make_tile_pil

        work = partial(mem_safe_make_tile, tile_dir, tile_f, array)
        jobs = tile_params
        OutputManager.set_description(pbar_ref, f"Converting {name}")
        OutputManager.set_units_total(pbar_ref, "tile", total_tiles)

        for job in jobs:
            if job[0] == max_zoom:
                work(job)
                OutputManager.update(pbar_ref, 1)
            else:
                break

        del array
        work = partial(mem_safe_make_tile, tile_dir, image_engine, None)
        work(job)
        OutputManager.update(pbar_ref, 1)

        for job in jobs:
            work(job)
            OutputManager.update(pbar_ref, 1)

    OutputManager.update_done(pbar_ref)


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
    raw_cols = list(
        map(
            lambda s: s.lower() if s in change_case else s,
            raw_col_vals,
        )
    )

    # if header line starts with a '#' exclude it
    if raw_cols[0] == "#":
        return raw_cols[1:]
    else:
        return raw_cols


def line_to_json(
    wcs: WCS,
    columns: List[str],
    catalog_assets_path: str,
    src_vals: List[str],
    catalog_starts_at_one: bool = 1,
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

        [[img_x, img_y]] = wcs.wcs_world2pix([[ra, dec]], int(catalog_starts_at_one))

    if "a" in columns and "b" in columns and "theta" in columns:
        a = float(src_vals[columns.index("a")])
        b = float(src_vals[columns.index("b")])
        b = a if np.isnan(b) else b

        theta = float(src_vals[columns.index("theta")])
        theta = 0 if np.isnan(theta) else theta

    else:
        a = -1
        b = -1
        theta = -1

    # Leaflet is 0 indexed, so if the catalog starts at 1 we need to offset
    # The convention for leaflet is to place markers at the lower left corner
    # of a pixel, to center the marker on the pixel, we subtract 1 to bring it
    # to leaflet convention and add 0.5 to move it to the center of the pixel.
    x = (img_x - int(catalog_starts_at_one)) + 0.5
    y = (img_y - int(catalog_starts_at_one)) + 0.5

    src_vals += [y, x, catalog_assets_path.split(os.sep)[-1]]

    src_json = os.path.join(catalog_assets_path, f"{src_id}.cbor")
    with open(src_json, "wb") as f:
        cbor2.dump(dict(id=src_id, v=src_vals), f)

    return dict(
        geometry=dict(
            coordinates=[x, y],
        ),
        tags=dict(
            a=a,
            b=b,
            theta=theta,
            catalog_id=src_id,
            cat_path=os.path.basename(catalog_assets_path),
        ),
    )


def process_catalog_file_chunk(
    process_f: Callable,
    fname: str,
    delimiter: str,
    q: queue.Queue,
    start: int,
    end: int,
) -> List[dict]:
    # newline="" for csv reader, see
    # https://docs.python.org/3/library/csv.html#csv.reader
    f = open(fname, "r", newline="")
    f.seek(start)
    f.readline()  # id start==0 skip cols, else advance to next complete line

    json_lines = []
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
            q.put(count)
            count = 1

    if count > 1:
        q.put(count)

    f.close()
    return json_lines


def _simplify_mixed_ws(catalog_fname: str) -> None:
    with open(catalog_fname, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    with open(catalog_fname, "w") as f:
        for line in lines:
            f.write(" ".join([token.strip() for token in line.split()]) + "\n")


def make_marker_tile(
    cluster: Supercluster,
    out_dir: str,
    job: Union[Tuple[int, Tuple[int, int]], List[Tuple[int, Tuple[int, int]]]],
) -> None:
    jobs = [job] if isinstance(job, tuple) else job

    for z, (y, x) in jobs:
        if not os.path.exists(os.path.join(out_dir, str(z))):
            os.mkdir(os.path.join(out_dir, str(z)))

        if not os.path.exists(os.path.join(out_dir, str(z), str(y))):
            os.mkdir(os.path.join(out_dir, str(z), str(y)))

        out_path = os.path.join(out_dir, str(z), str(y), f"{x}.pbf")

        tile_sources = cluster.get_tile(z, x, y)

        if tile_sources:
            tile_sources["name"] = "Points"

            for i in range(len(tile_sources["features"])):
                tile_sources["features"][i][
                    "geometry"
                ] = "POINT(0 0)"  # we dont' use this

            encoded_tile = mvt.encode([tile_sources], extents=256)

            with open(out_path, "wb") as f:
                f.write(encoded_tile)


def tile_markers(
    wcs_file: str,
    out_dir: str,
    catalog_delim: str,
    mp_procs: int,
    prefer_xy: bool,
    cluster_min_points: int,
    cluster_radius: float,
    cluster_node_size: int,
    min_zoom: int,
    max_zoom: int,
    tile_size: int,
    max_x: int,
    max_y: int,
    catalog_starts_at_one: bool,
    catalog_file: str,
    pbar_ref: Tuple[int, queue.Queue],
    batch_size: int = 500,
) -> None:
    _, fname = os.path.split(catalog_file)

    catalog_layer_name = get_map_layer_name(catalog_file)
    if catalog_layer_name in os.listdir(out_dir):
        OutputManager.write(pbar_ref, f"{fname} already tiled. Skipping tiling.")
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

    assert ra_dec_coords or x_y_coords, (
        catalog_file + " is missing coordinate columns (ra/dec, xy),"
    )
    assert "id" in columns, catalog_file + " missing 'id' column"
    assert use_xy or (wcs_file is not None), (
        catalog_file + " uses ra/dec coords, but a WCS file wasn't provided."
    )

    wcs = WCS(wcs_file) if wcs_file else None

    with open(os.path.join(out_dir, f"{catalog_layer_name}.columns"), "w") as f:
        f.write(",".join(columns))

    catalog_assets_parent_path = os.path.join(out_dir, "catalog_assets")
    if "catalog_assets" not in os.listdir(out_dir):
        os.mkdir(catalog_assets_parent_path)

    catalog_assets_path = os.path.join(catalog_assets_parent_path, catalog_layer_name)
    if catalog_layer_name not in os.listdir(catalog_assets_parent_path):
        os.mkdir(catalog_assets_path)

    line_func = partial(
        line_to_json,
        wcs,
        columns,
        catalog_assets_path,
        catalog_starts_at_one=catalog_starts_at_one,
    )

    OutputManager.set_description(pbar_ref, f"Parsing {catalog_file}")

    catalog_file_size = os.path.getsize(catalog_file)

    q = queue.Queue()
    process_f = partial(
        process_catalog_file_chunk, line_func, catalog_file, catalog_delim, q
    )

    if mp_procs > 1:
        process_f = ray.remote(num_cpus=1)(process_catalog_file_chunk)
        boundaries = np.linspace(0, catalog_file_size, mp_procs + 1, dtype=np.int64)

        remote_val_ids = list(
            starmap(
                process_f.remote,
                zip(
                    repeat(line_func),
                    repeat(catalog_file),
                    repeat(catalog_delim),
                    repeat(q),
                    boundaries[:-1],
                    boundaries[1:],
                ),
            )
        )

        _, remaining = ray.wait(remote_val_ids, timeout=0.01, fetch_local=False)
        while remaining:
            try:
                lines_processed = sum([q.get(timeout=0.0001) for _ in range(mp_procs)])
                OutputManager.update(pbar_ref, lines_processed)
            except queue.Empty:
                pass
            _, remaining = ray.wait(remaining, timeout=0.01, fetch_local=False)
        q.shutdown()
        catalog_values = list(chain.from_iterable(list(ray.get(remote_val_ids))))
        del q
    else:
        catalog_values = process_f(0, catalog_file_size)
        q.shutdown()
        del q

    OutputManager.set_description(
        pbar_ref, f"Clustering {catalog_file}(THIS MAY TAKE A WHILE)"
    )
    OutputManager.set_units_total(
        pbar_ref, unit="zoom levels", total=max_zoom + 1 - min_zoom
    )

    if cluster_radius is None:
        cluster_radius = max(max(max_x, max_y) / tile_size, 40)
    if cluster_node_size is None:
        cluster_node_size = np.log2(len(catalog_values)) * 2

    # cluster the parsed sources
    # need to get super cluster stuff in here
    clusterer = Supercluster(
        min_zoom=min_zoom,
        max_zoom=max_zoom - 1,
        min_points=cluster_min_points,
        extent=tile_size,
        radius=cluster_radius,
        node_size=cluster_node_size,
        alternate_CRS=(max_x, max_y),
        update_f=lambda: OutputManager.update(pbar_ref, 1),
        log=True,
    ).load(catalog_values)

    # Don't push any updates from the clustering algorithm after clustering
    # has been completed
    clusterer.update_f = None
    clusterer.log = False

    # tile the sources and save using protobuf
    zs = range(min_zoom, max_zoom + 1)
    ys = [range(2**z) for z in zs]
    xs = [range(2**z) for z in zs]
    tile_idxs = list(
        chain.from_iterable(
            [zip(repeat(zs[i]), product(ys[i], xs[i])) for i in range(len(zs))]
        )
    )

    catalog_layer_dir = os.path.join(out_dir, catalog_layer_name)

    OutputManager.set_description(pbar_ref, f"Tiling {catalog_file}")
    OutputManager.set_units_total(pbar_ref, unit="tile", total=len(tile_idxs))

    if mp_procs > 1:
        # We need to process batches to offset the cost of spinning up a process
        def batch_params(iter, batch_size):
            while True:
                batch = list(islice(iter, batch_size))
                if batch:
                    yield batch
                else:
                    break

        tile_f = ray.remote(num_cpus=1)(make_marker_tile)
        cluster_remote_id = ray.put(clusterer)

        utils.backpressure_queue_ray(
            tile_f.remote,
            list(
                zip(
                    repeat(cluster_remote_id),
                    repeat(catalog_layer_dir),
                    batch_params(iter(tile_idxs), batch_size),
                )
            ),
            pbar_ref,
            mp_procs,
            batch_size,
        )
    else:
        for zyx in tile_idxs:
            make_marker_tile(clusterer, catalog_layer_dir, zyx)
            OutputManager.update(pbar_ref, 1)


def files_to_map(
    files: List[str],
    out_dir: str = ".",
    title: str = "FitsMap",
    task_procs: int = 0,
    procs_per_task: int = 0,
    catalog_delim: str = ",",
    cat_wcs_fits_file: str = None,
    max_catalog_zoom: int = -1,
    tile_size: Tuple[int, int] = [256, 256],
    norm_kwargs: dict = {},
    rows_per_column: int = np.inf,
    prefer_xy: bool = False,
    catalog_starts_at_one: bool = True,
    img_tile_batch_size: int = 1000,
    cluster_min_points: int = 2,
    cluster_radius: float = None,
    cluster_node_size: int = None,
) -> None:
    """Converts a list of files into a LeafletJS map.

    Args:
        files (List[str]): List of files to convert into a map, can include image
                           files (.fits, .png, .jpg) and catalog files (.cat)
        out_dir (str): Directory to place the genreated web page and associated
                       subdirectories
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
        norm_kwargs (Union[Dict[str, Any], Dict[str, Dict[str, Any]]]):
                            Optional normalization keyword arguments passed to
                            `astropy.visualization.simple_norm`. Can either be
                            a single dictionary of keyword arguments, or a
                            dictionary of keyword arguments for each image
                            where the keys are the image names not full paths.
                            The default is linear scaling using min/max values.
                            See documentation for more information:
                            https://docs.astropy.org/en/stable/api/astropy.visualization.mpl_normalize.simple_norm.html
        rows_per_column (int): If converting a catalog, the number of items in
                               have in each column of the marker popup.
                               By default produces all values in a single
                               column. Setting this value can make it easier to
                               work with catalogs that have a lot of values for
                               each object.
        prefer_xy (bool): If True x/y coordinates should be preferred if both
                          ra/dec and x/y are present in a catalog
        catalog_starts_at_one (bool): True if the catalog is 1 indexed, False if
                                      the catalog is 0 indexed
        img_tile_batch_size (int): The number of image tiles to process in
                                   parallel when task_procs > 1
        cluster_min_points (int): The minimum points to form a catalog cluster
        cluster_radius (float): The radius of each cluster in pixels.
        cluster_node_size (int): The size for the kd-tree leaf mode, afftects performance.

    Example of image specific norm_kwargs vs single norm_kwargs:

    >>> norm_kwargs = {
    >>>    "test.fits": {"stretch": "log", "min_percent": 1, "max_percent": 99.5},
    >>>    "test2.fits": {"stretch": "linear", "min_percent": 5, "max_percent": 99.5},
    >>> }
    >>> # or
    >>> norm_kwargs = {"stretch": "log", "min_percent": 1, "max_percent": 99.5}


    Returns:
        None
    """

    assert len(files) > 0, "No files provided `files` is an empty list"

    unlocatable_files = list(filterfalse(os.path.exists, files))
    assert len(unlocatable_files) == 0, f"Files not found:{unlocatable_files}"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if "js" not in os.listdir(out_dir):
        os.mkdir(os.path.join(out_dir, "js"))

    if "css" not in os.listdir(out_dir):
        os.mkdir(os.path.join(out_dir, "css"))

    # build image tasks
    img_files = filter_on_extension(files, IMG_FORMATS)
    img_layer_names = list(map(get_map_layer_name, img_files))

    img_f_kwargs = dict(
        tile_size=tile_size,
        out_dir=out_dir,
        mp_procs=procs_per_task,
        norm_kwargs=norm_kwargs,
        batch_size=img_tile_batch_size,
    )

    # we want to init ray in such a way that it doesn't print any output
    # to the console. These should be changed during development
    debug = os.getenv("FITSMAP_DEBUG", "False").lower() == "true"
    ray.init(
        include_dashboard=debug,  # during dev == True
        configure_logging=~debug,  # during dev == False
        logging_level=(
            logging.INFO if debug else logging.CRITICAL
        ),  # during dev == logging.INFO, test == logging.CRITICAL
        log_to_driver=debug,  # during dev = True
    )

    if task_procs > 1:
        img_task_f = ray.remote(num_cpus=1)(tile_img).remote
    else:
        img_task_f = tile_img

    # build catalog tasks
    cat_files = filter_on_extension(files, CAT_FORMAT)
    cat_layer_names = list(map(get_map_layer_name, cat_files))

    max_x, max_y = utils.peek_image_info(img_files)
    max_dim = max(max_x, max_y)
    if len(cat_files) > 0:
        # get highlevel image info for catalogging function
        max_zoom = int(np.log2(2 ** np.ceil(np.log2(max_dim)) / tile_size[0]))
        max_dim = 2**max_zoom * tile_size[0]
        if max_catalog_zoom == -1:
            max_zoom = int(np.log2(2 ** np.ceil(np.log2(max_dim)) / tile_size[0]))
        else:
            max_zoom = max_catalog_zoom

        if task_procs > 1:
            cat_task_f = ray.remote(num_cpus=1)(tile_markers).remote
        else:
            cat_task_f = tile_markers

        cat_f_kwargs = dict(
            wcs_file=cat_wcs_fits_file,
            out_dir=out_dir,
            catalog_delim=catalog_delim,
            mp_procs=procs_per_task,
            prefer_xy=prefer_xy,
            min_zoom=0,
            max_zoom=max_zoom,
            tile_size=tile_size[0],
            max_x=max_dim,
            max_y=max_dim,
            catalog_starts_at_one=catalog_starts_at_one,
            cluster_min_points=cluster_min_points,
            cluster_radius=cluster_radius,
            cluster_node_size=cluster_node_size,
        )
    else:
        cat_task_f = None
        cat_f_kwargs = dict()

    output_manager = OutputManager()

    img_tasks = zip(
        repeat(img_task_f),
        map(
            lambda x: dict(**x[0], **x[1]),
            zip(
                repeat(img_f_kwargs),
                starmap(
                    lambda x, y: dict(file_location=x, pbar_ref=y),
                    zip(img_files, output_manager.make_bar()),
                ),
            ),
        ),
    )

    cat_tasks = zip(
        repeat(cat_task_f),
        map(
            lambda x: dict(**x[0], **x[1]),
            zip(
                repeat(cat_f_kwargs),
                starmap(
                    lambda x, y: dict(catalog_file=x, pbar_ref=y),
                    zip(cat_files, output_manager.make_bar()),
                ),
            ),
        ),
    )

    tasks = chain(img_tasks, cat_tasks)

    if task_procs > 1:
        # start runnning task_procs number of tasks
        in_progress = list(
            starmap(
                lambda i, func_kwargs: func_kwargs[0](**func_kwargs[1]),
                zip(range(task_procs), tasks),
            )
        )
        while in_progress:
            _, in_progress = ray.wait(in_progress, timeout=0.003)
            output_manager.check_for_updates()
            if len(in_progress) < task_procs:
                try:
                    # try to get a task with kwargs from the iterator
                    func, kwargs = next(tasks)
                    in_progress.append(func(**kwargs))
                    print(in_progress)
                except StopIteration:
                    # all of the tasks are in progress or completed
                    if not in_progress:
                        # we're done!
                        break
                    else:
                        # the tasks 'in_progress' are still running
                        pass
    else:

        def f(func_args):
            func_args[0](**func_args[1])

        any(map(f, tasks))

    output_manager.close_up()
    ray.shutdown()
    print("Building index.html")

    cat_wcs = WCS(cat_wcs_fits_file) if cat_wcs_fits_file else None

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
    title: str = "FitsMap",
    task_procs: int = 0,
    procs_per_task: int = 0,
    catalog_delim: str = ",",
    cat_wcs_fits_file: str = None,
    max_catalog_zoom: int = -1,
    tile_size: Shape = [256, 256],
    norm_kwargs: Union[Dict[str, Any], Dict[str, Dict[str, Any]]] = {},
    rows_per_column: int = np.inf,
    prefer_xy: bool = False,
    catalog_starts_at_one: bool = True,
    img_tile_batch_size: int = 1000,
    cluster_min_points: int = 2,
    cluster_radius: float = None,
    cluster_node_size: int = None,
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
        norm_kwargs (Union[Dict[str, Any], Dict[str, Dict[str, Any]]]):
                            Optional normalization keyword arguments passed to
                            `astropy.visualization.simple_norm`. Can either be
                            a single dictionary of keyword arguments, or a
                            dictionary of keyword arguments for each image
                            where the keys are the image names not full paths.
                            The default is linear scaling using min/max values.
                            See documentation for more information:
                            https://docs.astropy.org/en/stable/api/astropy.visualization.mpl_normalize.simple_norm.html
        rows_per_column (int): If converting a catalog, the number of items in
                               have in each column of the marker popup.
                               By default produces all values in a single
                               column. Setting this value can make it easier to
                               work with catalogs that have a lot of values for
                               each object.
        prefer_xy (bool): If True x/y coordinates should be preferred if both
                          ra/dec and x/y are present in a catalog
        catalog_starts_at_one (bool): True if the catalog is 1 indexed, False if
                                      the catalog is 0 indexed
        img_tile_batch_size (int): The number of image tiles to process in
                                   parallel when task_procs > 1
        cluster_min_points (int): The minimum points to form a catalog cluster
        cluster_radius (float): The radius of each cluster in pixels.
        cluster_node_size (int): The size for the kd-tree leaf mode, afftects performance.

    Example of image specific norm_kwargs vs single norm_kwargs:

    >>> norm_kwargs = {
    >>>    "test.fits": {"stretch": "log", "min_percent": 1, "max_percent": 99.5},
    >>>    "test2.fits": {"stretch": "linear", "min_percent": 5, "max_percent": 99.5},
    >>> }
    >>> # or
    >>> norm_kwargs = {"stretch": "log", "min_percent": 1, "max_percent": 99.5}

    Returns:
        None

    Raises:
        ValueError if the dir is empty, there are no convertable files or if
        ``exclude_predicate`` exlcudes all files
    """
    dir_files = list(
        map(
            lambda d: os.path.join(directory, d),
            filterfalse(
                exclude_predicate,
                os.listdir(directory),
            ),
        )
    )

    assert (
        len(dir_files) > 0
    ), "No files in `directory` or `exclude_predicate` excludes everything"

    files_to_map(
        sorted(dir_files),
        out_dir=out_dir,
        title=title,
        task_procs=task_procs,
        procs_per_task=procs_per_task,
        catalog_delim=catalog_delim,
        cat_wcs_fits_file=cat_wcs_fits_file,
        max_catalog_zoom=max_catalog_zoom,
        tile_size=tile_size,
        norm_kwargs=norm_kwargs,
        rows_per_column=rows_per_column,
        prefer_xy=prefer_xy,
        catalog_starts_at_one=catalog_starts_at_one,
        img_tile_batch_size=img_tile_batch_size,
        cluster_min_points=cluster_min_points,
        cluster_radius=cluster_radius,
        cluster_node_size=cluster_node_size,
    )
