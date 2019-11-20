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

import json
import os
import shutil
import sys
from functools import partial, reduce
from itertools import chain, count, filterfalse, product, repeat
from multiprocessing import Pool
from typing import Callable, List, Tuple, Union

import sharedmem
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from imageio import imread
# https://github.com/zimeon/iiif/issues/11#issuecomment-131129062
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = sys.maxsize


Shape = Tuple[int, int]

IMG_FORMATS = ["fits", "jpg", "png"]
CAT_FORMAT = ["cat"]
IMG_ENGINE_PIL = "PIL"
IMG_ENGINE_MPL = "MPL"


def dir_to_map(
    directory: str,
    tile_size: Shape = [256, 256],
    depth: int = None,
    method: str = METHOD_RECURSIVE,
    image_engine: str = convert.IMG_ENGINE_MPL,
    title: str = "FitsMap",
    multiprocessing_processes: int = 0,
    cat_wcs_fits_file: str = None,
    out_dir: str = None,
):
    if out_dir:
        _map = Map(out_dir, title)
    else:
        _map = Map(directory, title)

    dir_entries = list(
        map(
            lambda d: os.path.join(directory, d),
            filter(
                lambda d: os.path.splitext(d)[1][1:] in IMG_FORMATS,
                os.listdir(directory),
            ),
        )
    )

    kwargs = dict(
        tile_size=tile_size,
        depth=depth,
        method=method,
        image_engine=image_engine,
        out_dir=out_dir,
    )
    layer_func = partial(img_to_layer, **kwargs)

    pbar_loc_generator = count(0)
    if multiprocessing_processes > 0:
        with Pool(multiprocessing_processes) as p:
            built_layers = p.starmap(layer_func, zip(dir_entries, pbar_loc_generator))
    else:
        built_layers = map(layer_func, dir_entries, pbar_loc_generator)

    def update_map(shp1: Tuple[int, int], layer: Tuple[str, str, Shape]) -> np.ndarray:
        p, n, shp2 = layer
        _map.add_tile_layer(p, n)
        return np.maximum(shp1, shp2)

    max_xy = reduce(update_map, built_layers, (0, 0))

    catalogs = list(
        map(
            lambda d: os.path.join(directory, d),
            filter(lambda d: d.endswith(".cat"), os.listdir(directory)),
        )
    )

    if len(catalogs) > 0:
        if cat_wcs_fits_file is None:
            err_msg = [
                "There are catalog(.cat) in files in " + directory + ", but no",
                "value was given for 'cat_wcs_fits_file' skipping catalog coversion",
            ]
            print(" ".join(err_msg))
        else:
            cat_func = partial(convert.catalog, cat_wcs_fits_file, out_dir, max_xy)

            if multiprocessing_processes > 0:
                with Pool(multiprocessing_processes) as p:
                    list(
                        map(
                            _map.add_marker_catalog,
                            list(
                                p.starmap(cat_func, zip(catalogs, pbar_loc_generator))
                            ),
                        )
                    )
            else:
                # unlist this maybe?
                list(
                    map(
                        _map.add_marker_catalog,
                        list(map(cat_func, catalogs, pbar_loc_generator)),
                    )
                )

    _map.build_map()

def build_path(depth, y, x, out_dir):
    z, y, x = str(z), str(y), str(x)
    z_dir = os.path.join(out_dir, z)
    y_dir = os.path.join(z_dir, y)

    img_path = os.path.join(y_dir, "{}.png".format(x))

    return img_path


def slice_idx_generator(shape:Tuple[int, int], zoom:int) -> Iterable[Tuple[int, int, int, slice, slice]]:
    num_splits = int((4 ** zoom) ** 0.5)
    start, end = 0, shape[0]
    splits = range(start, end+1, end//num_splits)
    rows = zip(range(num_splits-1, -1, -1), zip(splits[:-1], splits[1:]))
    cols = enumerate(zip(splits[:-1], splits[1:]))
    rows_cols = product(rows, cols)

    def transform_iteration(row_col):
        ((y, (start_y, end_y)), (x, (start_x, end_x))) = row_col
        return (zoom, y, x, slice(start_y, end_y), slice(start_x, end_x))

    return map(transform_iteration, rows_cols)

def get_array(file_location: str) -> np.ndarray:
    _, ext = os.path.splitext(file_location)

    if ext == "fits":
        array = fits.getdata(file_location)
        shape = array.shape
        if len(shape) > 2:
            raise ValueError("FitsMap only supports 2D FITS files.")
    else:
        array = np.flipud(imread(file_location))

        if len(array.shape) == 3:
            shape = array.shape[:-1]
        elif len(array.shape) == 2:
            shape = array.shape
        else:
            raise ValueError("FitsMap only supports 2D and 3D images.")

    if shape[0] != shape[1]:
        raise ValueError("Only square images are currently supported")

    return array


def filter_on_extension(
    files: List[str], extensions: List[str], exclude_predicate: Callable = None
) -> List[str]:
    neg_predicate = exclude_predicate if exclude_predicate else lambda x: False

    return list(
        filter(
            lambda s: (os.path.splitext(s)[1][1:] in extensions)
            and not neg_predicate(s),
            files,
        )
    )


def make_dirs(out_dir, zoom):
    def sub_dir(f):
        try:
            os.makedirs(os.path.join(out_dir, f))
        except FileExistsError:
            pass

    def build_z_ys(z, ys):
        list(map(lambda y: sub_dir(f"{z}/{y}"), ys))

    def build_zs(z):
        ys = range(int(np.sqrt(4 ** z)))
        build_z_ys(z, ys)

    zs = range(zoom + 1)
    list(map(build_zs, zs))

def make_tile(array:np.ndarray, vmin:float, vmax:float, out_dir:str, img_engine:str, job:Tuple[int, int, int, slice, slice]):
    z, y, x, slice_ys, slice_xs = job

    img_path = build_path(z, y, x, out_dir)

    if img_engine==IMG_ENGINE_MPL:
        f = plt.figure(dpi=100)
        f.set_size_inches([256 / 100, 256 / 100])
        plt.imshow(
            array[slice_ys, slice_xs],
            origin="lower",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        # plt.text(array.shape[0]//2, array.shape[1]//2, f"{depth},{y},{x}")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis("off")
        plt.savefig(img_path, dpi=100, bbox_inches=0, interpolation="nearest")
        plt.close(f)
    else:
        img = Image.fromarray(array[slice_ys, slice_xs])
        img.thumbnail([256, 256], Image.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(img_path, "PNG")
        del img


def tile_img(
    file_location: str,
    pbar_loc: int,
    tile_size: Shape = [256, 256],
    zoom: int = None,
    image_engine: str = convert.IMG_ENGINE_PIL,
    out_dir: str = ".",
    mp_procs: int = 0,
) -> None:
    # get image
    array = get_array(file_location)
    arr_min, arr_max = array.min(), array.max()

    z = zoom if zoom else int(np.log2(array.shape[0] / tile_size[0]))

    # build directory structure
    name = get_map_layer_name(file_location)
    tile_dir = os.path.join(out_dir, name)
    if name not in os.listdir(out_dir):
        os.mkdir(tile_dir)

    make_dirs(tile_dir, z)

    # tile the image
    total_tiles = sum([4**i for i in range(z+1)])
    pbar = tqdm(position=pbar_loc, desc="Converting " + name, unit="tile", total=total_tiles)

    tile_params = chain.from_iterable([slice_idx_generator(array.shape, i) for i in range(z+1)])

    if mp_procs:
        mp_array = sharedmem.empty_like(array)
        mp_array[...] = array[...]
        work = partial(make_tile, array, arr_min, arr_max, tile_dir, image_engine)
        with Pool(mp_procs) as p:
            any(p.imap_unordered(work, tqdm(tile_params, desc="tiles", total=total_tiles)))
    else:
        work = partial(make_tile, array, arr_min, arr_max, tile_dir, image_engine)
        any(map(work, tqdm(tile_params, desc="Converting " + name, total=total_tiles, unit="tile")))


def get_map_layer_name(file_location: str):
    _, fname = os.path.split(file_location)
    name = os.path.splitext(fname)[0].replace(".", "_").replace("-", "_")
    return name

def line_to_cols(raw_line: str):

    change_case = ["RA", "DEC", "Ra", "Dec"]

    # make ra and dec lowercase for ease of access
    raw_cols = list(
        map(lambda s: s.lower() if s in change_case else s, raw_line.strip().split())
    )

    # if header line starts with a '#' exclude it
    if raw_cols[0] == "#":
        return raw_cols[1:]
    else:
        return raw_cols


def line_to_json(wcs: WCS, columns: List[str], max_dim: int, src_line: str):
    src_vals = src_line.strip().split()

    ra = float(src_vals[columns.index("ra")])
    dec = float(src_vals[columns.index("dec")])
    src_id = int(src_vals[columns.index("id")])

    [[img_x, img_y]] = wcs.wcs_world2pix([[ra, dec]], 0)

    x = img_x / max_dim * 256
    y = img_y / max_dim * 256 - 256

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
):
    wcs = WCS(wcs_file)

    f = open(catalog_file, "r")

    columns = line_to_cols(next(f))

    if "ra" not in columns or "dec" not in columns or "id" not in columns:
        err_msg = " ".join(
            [
                catalog_file + " is missing an 'ra' column, a 'dec' column,",
                "an 'id' column, or all of the above",
            ]
        )
        raise ValueError(err_msg)

    header = fits.getheader(wcs_file)
    max_dim = max(header["NAXIS1"], header["NAXIS2"])

    line_func = partial(line_to_json, wcs, columns, max_dim)

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

    return cat_file

def files_to_map(
    files: List[str],
    out_dir: str = ".",
    # exclude_predicate:Callable = None, (put this in dir to map)
    zoom: int = None,
    title: str = "FitsMap",
    job_procs: int = 0,
    procs_per_job: int = 0,
    cat_wcs_fits_file: str = None,
    tile_size: Shape = [256, 256],
    image_engine: str = convert.IMG_ENGINE_PIL,
):
    img_f_kwargs = dict(
        tile_size=tile_size, zoom=zoom, image_engine=image_engine, out_dir=out_dir,
    )

    img_files = filter_on_extension(files, IMG_FORMATS)
    map_layer_names = list(map(get_map_layer_name, img_files))
    img_job_f = partial(tile_img, **img_f_kwargs)

    cat_files = filter_on_extension(files, CAT_FORMAT)

    if len(cat_files) > 0:
        if cat_wcs_fits_file is None:
            err_msg = [
                "Catalog files (.cat) were included, but no value was given for"
                "`cat_wcs_fits_file`. ra/dec can't be converted without a fits"
                "file. Skipping catalog conversion"
            ]
            print(" ".join(err_msg))


        cat_func = partial(catalog_to_markers, cat_wcs_fits_file, out_dir)

        # finish catalogging

    # combine cat and img jobs

    # execute sync or async


    pbar_locations = count(0)
