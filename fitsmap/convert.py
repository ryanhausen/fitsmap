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
from functools import partial
import shutil
from typing import List, Tuple, Union


import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image
from tqdm import tqdm

IMG_ENGINE_PIL = "PIL"
IMG_ENGINE_MPL = "MPL"

# TODO: Try to understand why leaflet prefers the coords like this
def _get_new_coords(y, x):
    adj = lambda z: 2 * z
    adj_y = adj(y)
    adj_x = adj(x)

    return [
        (adj_y + 1, adj_x),
        (adj_y + 1, adj_x + 1),
        (adj_y, adj_x),
        (adj_y, adj_x + 1),
    ]


def _get_depth(shape, tile_size):
    return int(np.log2(shape[0]))


def _build_path(depth, y, x, out_dir):
    depth, y, x = str(depth), str(y), str(x)

    z_dir = os.path.join(out_dir, depth)

    if depth not in os.listdir(out_dir):
        os.mkdir(z_dir)

    y_dir = os.path.join(z_dir, y)
    if y not in os.listdir(z_dir):
        os.mkdir(y_dir)

    img_path = os.path.join(y_dir, "{}.png".format(x))

    return img_path


def _convert_and_save(array, depth, y, x, out_dir, tile_size, image_engine, vmin, vmax):
    path = _build_path(depth, y, x, out_dir)

    if image_engine == "PIL":
        img = Image.fromarray(array)
        img.thumbnail(tile_size)
        if img.mode != "RGB":
            img = img.convert("RGB")

        img.save(path, "PNG")
        del img
    else:
        f = plt.figure(dpi=100)
        f.set_size_inches([tile_size[0] / 100, tile_size[1] / 100])
        plt.imshow(
            array,
            origin="lower",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        # plt.text(array.shape[0]//2, array.shape[1]//2, f"{depth},{y},{x}")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis("off")
        plt.savefig(path, dpi=100, bbox_inches=0, interpolation="nearest")
        plt.close(f)


def array(
    array,
    pbar,
    tile_size=[256, 256],
    depth=None,
    out_dir=".",
    method="recursive",
    image_engine="mpl",
):
    if method == "recursive":
        if depth is None:
            depth = _get_depth(array.shape, tile_size)

    if method == "recursive":

        total_tiles = (1 - (4 ** (depth + 1))) // (1 - 4)

        pbar.total = total_tiles

        _build_recursively(
            array,
            (0, 0),
            0,
            depth,
            out_dir,
            tile_size,
            image_engine,
            array.min(),
            array.max(),
            pbar,
        )

        pbar.close()
    elif method == "iterative":
        raise NotImplementedError("iterative not supported")
    else:
        raise ValueError("{} invalid. Please use recursive".format(method))


def _build_recursively(
    array, coords, depth, goal, out_dir, tile_size, image_engine, vmin, vmax, pbar
):
    y, x = coords

    _convert_and_save(array, depth, y, x, out_dir, tile_size, image_engine, vmin, vmax)
    pbar.update()

    if depth < goal:
        ax0, ax1 = array.shape[0], array.shape[1]

        split_0 = ax0 // 2
        split_1 = ax1 // 2

        slices = [
            (slice(0, split_0), slice(0, split_1)),
            (slice(0, split_0), slice(split_1, ax1)),
            (slice(split_0, ax0), slice(0, split_1)),
            (slice(split_0, ax0), slice(split_1, ax1)),
        ]

        for (_ys, _xs), crds in zip(slices, _get_new_coords(y, x)):
            arr = array[_ys, _xs]
            _build_recursively(
                arr,
                crds,
                depth + 1,
                goal,
                out_dir,
                tile_size,
                image_engine,
                vmin,
                vmax,
                pbar,
            )
            del arr


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


def line_to_json(wcs: WCS, columns: List[str], max_xy: Tuple[int, int], src_line: str):
    max_x, max_y = max_xy

    src_vals = src_line.strip().split()

    ra = float(src_vals[columns.index("ra")])
    dec = float(src_vals[columns.index("dec")])
    src_id = int(src_vals[columns.index("id")])

    [[img_x, img_y]] = wcs.wcs_world2pix([[ra, dec]], 0)

    x = img_x / max_x * 256
    y = img_y / max_y * 256 - 256

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


def catalog(
    wcs_file: str,
    out_dir: str,
    max_xy: Tuple[int, int],
    catalog_file: str,
    pbar_loc: int,
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

    line_func = partial(line_to_json, wcs, columns, max_xy)

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
