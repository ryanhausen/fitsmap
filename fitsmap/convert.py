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


def catalog(
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
