# MIT License
# Copyright 2021 Ryan Hausen and contributors

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

"""Helper functions for creating a leaflet JS HTML map."""

# ******************************************************************************
# Designed for internal use. Any method/variable can be deprecated/changed
# without consideration.
# ******************************************************************************

import os
import shutil
from itertools import count, repeat, starmap
from functools import partial, reduce
from typing import Dict, List, Tuple

import numpy as np
from astropy.wcs import WCS

import fitsmap.utils as utils

LAYER_ATTRIBUTION = "<a href='https://github.com/ryanhausen/fitsmap'>FitsMap</a>"


def chart(
    out_dir: str,
    title: str,
    img_layer_names: List[str],
    marker_layer_names: List[str],
    wcs: WCS,
    rows_per_column: int,
    max_xy: Tuple[int, int],
) -> None:
    """Creates an HTML file containing a leaflet js map using the given params.

    ****************************************************************************
    * Designed for internal use. Any method/variable can be deprecated/changed *
    * without consideration.                                                   *
    ****************************************************************************
    """
    # convert layer names into a single javascript string
    layer_zooms = lambda l: list(map(int, os.listdir(os.path.join(out_dir, l))))
    zooms = reduce(lambda x, y: x + y, list(map(layer_zooms, img_layer_names)))
    zooms = [0] if len(zooms) == 0 else zooms
    convert_layer_name_func = partial(layer_name_to_dict, min(zooms), max(zooms))
    img_layer_dicts = list(
        starmap(convert_layer_name_func, zip(img_layer_names, repeat(None)))
    )
    cat_layer_dicts = list(
        starmap(convert_layer_name_func, zip(marker_layer_names, get_colors()))
    )

    # generated javascript =====================================================
    with open(os.path.join(out_dir, "js", "urlCoords.js"), "w") as f:
        f.write(build_urlCoords_js(wcs))

    with open(os.path.join(out_dir, "js", "index.js"), "w") as f:
        f.write(
            build_index_js(img_layer_dicts, cat_layer_dicts, rows_per_column, max_xy)
        )
    # generated javascript =====================================================

    # HTML file contents =======================================================
    extra_js = build_conditional_js(out_dir) if cat_layer_dicts else ""

    extra_css = build_conditional_css(out_dir)

    move_support_images(out_dir)

    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(build_html(title, extra_js, extra_css))
    # HTML file contents =======================================================


def layer_name_to_dict(min_zoom: int, max_zoom: int, name: str, color: str,) -> dict:
    """Convert layer name to dict for conversion."""

    layer_dict = dict(
        directory=name + "/{z}/{y}/{x}.png",
        name=name,
        min_zoom=min_zoom,
        max_zoom=max_zoom + 5,
        max_native_zoom=max_zoom,
    )
    if color:
        layer_dict["color"] = color
        layer_dict["directory"] = layer_dict["directory"].replace("png", "pbf")

    return layer_dict


def img_layer_dict_to_str(layer: dict) -> str:
    """Convert layer dict to layer str for including in HTML file."""

    layer_str = [
        "const " + layer["name"],
        ' = L.tileLayer("' + layer["directory"] + '"',
        ", { ",
        'attribution:"' + LAYER_ATTRIBUTION + '", ',
        "minZoom: " + str(layer["min_zoom"]) + ", ",
        "maxZoom: " + str(layer["max_zoom"]) + ", ",
        "maxNativeZoom: " + str(layer["max_native_zoom"]) + " ",
        "});",
    ]

    return "".join(layer_str)


def cat_layer_dict_to_str(layer: dict, rows_per_column: int) -> str:
    """Convert layer dict to layer str for including in HTML file."""

    rpc_str = "Infinity" if np.isinf(rows_per_column) else str(rows_per_column)
    layer_str = [
        "const " + layer["name"],
        " = L.gridLayer.tiledMarkers(",
        "{ ",
        'tileURL:"' + layer["directory"] + '", ',
        'color: "' + layer["color"] + '", ',
        f"rowsPerColumn: {rpc_str}, ",
        "minZoom: " + str(layer["min_zoom"]) + ", ",
        "maxZoom: " + str(layer["max_zoom"]) + ", ",
        "maxNativeZoom: " + str(layer["max_native_zoom"]) + " ",
        "});",
    ]

    return "".join(layer_str)


def get_colors() -> List[str]:
    return [
        "#4C72B0",
        "#DD8452",
        "#55A868",
        "#C44E52",
        "#8172B3",
        "#937860",
        "#DA8BC3",
        "#8C8C8C",
        "#CCB974",
        "#64B5CD",
    ]


def leaflet_crs_js(tile_layers: List[dict]) -> str:
    max_zoom = max(map(lambda t: t["max_native_zoom"], tile_layers))

    scale_factor = int(2 ** max_zoom)

    js = [
        "L.CRS.FitsMap = L.extend({}, L.CRS.Simple, {",
        f"    transformation: new L.Transformation(1/{scale_factor}, 0, -1/{scale_factor}, 256)",
        "});",
    ]

    return "\n".join(js)


def leaflet_map_js(tile_layers: List[dict]):
    js = "\n".join(
        [
            'const map = L.map("map", {',
            "    crs: L.CRS.FitsMap,",
            "    minZoom: " + str(max(map(lambda t: t["min_zoom"], tile_layers))) + ",",
            "    preferCanvas: true,",
            f"    layers: [{tile_layers[0]['name']}]",
            "});",
        ]
    )

    return js


def move_support_images(out_dir: str) -> List[str]:
    img_extensions = [".png", ".jpg", ".ico"]

    support_dir = os.path.join(os.path.dirname(__file__), "support")
    out_img_dir = os.path.join(out_dir, "imgs")

    local_img_files = list(
        filter(
            lambda f: os.path.splitext(f)[1] in img_extensions,
            sorted(os.listdir(support_dir)),
        )
    )

    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)

    all(
        map(
            lambda f: shutil.copy2(
                os.path.join(support_dir, f), os.path.join(out_img_dir, f)
            ),
            local_img_files,
        )
    )
    return local_img_files


def build_conditional_css(out_dir: str) -> str:

    search_css = "https://unpkg.com/leaflet-search@2.9.8/dist/leaflet-search.src.css"
    css_string = "    <link rel='stylesheet' href='{}'/>"

    support_dir = os.path.join(os.path.dirname(__file__), "support")
    out_css_dir = os.path.join(out_dir, "css")

    local_css_files = list(
        filter(
            lambda f: os.path.splitext(f)[1] == ".css", sorted(os.listdir(support_dir))
        )
    )

    if not os.path.exists(out_css_dir):
        os.mkdir(out_css_dir)

    all(
        map(
            lambda f: shutil.copy2(
                os.path.join(support_dir, f), os.path.join(out_css_dir, f)
            ),
            local_css_files,
        )
    )

    local_css = list(map(lambda f: f"css/{f}", local_css_files))

    return "\n".join(map(lambda s: css_string.format(s), [search_css] + local_css))


def build_conditional_js(out_dir: str) -> str:

    support_dir = os.path.join(os.path.dirname(__file__), "support")
    out_js_dir = os.path.join(out_dir, "js")

    local_js_files = list(
        sorted(
            filter(lambda f: os.path.splitext(f)[1] == ".js", os.listdir(support_dir))
        )
    )

    if not os.path.exists(out_js_dir):
        os.mkdir(out_js_dir)

    all(
        map(
            lambda f: shutil.copy2(
                os.path.join(support_dir, f), os.path.join(out_js_dir, f)
            ),
            local_js_files,
        )
    )

    remote_js = [
        '    <script src="https://unpkg.com/pbf@3.0.5/dist/pbf.js", crossorigin=""></script>',
        '    <script src="https://cdn.jsdelivr.net/npm/leaflet-search" crossorigin=""></script>',
    ]

    js_string = "    <script src='js/{}'></script>"
    local_js = list(map(lambda s: js_string.format(s), local_js_files))

    return "\n".join(remote_js + local_js)


def leaflet_layer_control_declaration(
    img_layer_dicts: List[Dict], cat_layer_dicts: List[Dict],
) -> str:
    img_layer_label_pairs = ",".join(
        list(map(lambda l: '"{0}":{0}'.format(l["name"]), img_layer_dicts))
    )

    cat_layer_label_pairs = ",".join(
        list(map(lambda l: '"{0}":{0}'.format(l["name"]), cat_layer_dicts))
    )

    control_js = [
        "const layerControl = L.control.layers(",
        f"    {{{img_layer_label_pairs}}},",
        f"    {{{cat_layer_label_pairs}}}",
        ").addTo(map);",
    ]

    return "\n".join(control_js)


def leaflet_search_control_declaration(cat_layer_dicts: List[Dict],) -> str:
    search_js = [
        "const catalogPaths = [",
        *list(
            map(
                lambda s: f'    "{os.path.join("catalog_assets", s)}/",',
                map(lambda l: l["name"], cat_layer_dicts),
            )
        ),
        "];",
        "",
        f"const searchControl = buildCustomSearch(catalogPaths, {cat_layer_dicts[0]['max_native_zoom']});",
        "map.addControl(searchControl);",
    ]

    return "\n".join(search_js) if cat_layer_dicts else ""


def build_urlCoords_js(img_wcs: WCS) -> str:

    wcs_js_file = os.path.join(os.path.dirname(__file__), "support", "urlCoords.js.tmp")

    with open(wcs_js_file, "r") as f:
        wcs_js = "".join(f.readlines())

    if img_wcs:
        wcs_js = wcs_js.replace("_IS_RA_DEC", "1")
        wcs_js = wcs_js.replace("_CRPIX", str(img_wcs.wcs.crpix.tolist()))
        wcs_js = wcs_js.replace("_CRVAL", str(img_wcs.wcs.crval.tolist()))

        if hasattr(img_wcs.wcs, "cd"):
            wcs_js = wcs_js.replace("_CD", str(img_wcs.wcs.cd.tolist()))
        else:
            # Manual "CD" matrix
            delta = img_wcs.all_pix2world(
                [
                    img_wcs.wcs.crpix,
                    img_wcs.wcs.crpix + np.array([1, 0]),
                    img_wcs.wcs.crpix + np.array([0, 1]),
                ],
                0,
            )

            _cd = np.array([delta[1, :] - delta[0, :], delta[2, :] - delta[0, :]])
            _cd[0, :] *= np.cos(img_wcs.wcs.crval[1] / 180 * np.pi)
            wcs_js = wcs_js.replace("_CD", str(_cd.tolist()))
    else:
        wcs_js = wcs_js.replace("_IS_RA_DEC", "0")
        wcs_js = wcs_js.replace("_CRPIX", "[1, 1]")
        wcs_js = wcs_js.replace("_CRVAL", "[0, 0]")
        wcs_js = wcs_js.replace("_CD", "[[1, 0], [0, 1]]")

    return wcs_js


def build_index_js(
    image_layer_dicts: List[Dict],
    marker_layer_dicts: List[str],
    rows_per_column: int,
    max_xy: Tuple[int, int],
) -> str:

    js = "\n".join(
        [
            "// Image layers ================================================================",
            *list(map(img_layer_dict_to_str, image_layer_dicts)),
            "",
            "// Marker layers ===============================================================",
            *list(
                starmap(
                    cat_layer_dict_to_str,
                    zip(marker_layer_dicts, repeat(rows_per_column)),
                )
            ),
            "",
            "// Basic map setup =============================================================",
            leaflet_crs_js(image_layer_dicts),
            "",
            leaflet_map_js(image_layer_dicts),
            "",
            leaflet_layer_control_declaration(image_layer_dicts, marker_layer_dicts),
            "",
            "// Search ======================================================================",
            leaflet_search_control_declaration(marker_layer_dicts)
            if len(marker_layer_dicts)
            else "",
            "",
            "// Map event setup =============================================================",
            'map.on("moveend", updateLocationBar);',
            'map.on("zoomend", updateLocationBar);',
            "",
            'if (urlParam("zoom")==null) {',
            f"    map.fitBounds(L.latLngBounds([[0, 0], [{max_xy[0]}, {max_xy[1]}]]));",
            "} else {",
            "    panFromUrl(map);",
            "}",
        ]
    )

    return js


def build_html(title: str, extra_js: str, extra_css: str) -> str:
    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <title>{}</title>".format(title),
        '    <meta charset="utf-8" />',
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        '    <link rel="shortcut icon" type="image/x-icon" href="imgs/favicon.ico" />',
        '    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.3.4/dist/leaflet.css" integrity="sha512-puBpdR0798OZvTTbP4A8Ix/l+A4dHDD0DGqYW6RQ+9jxkRFclaxxQb/SJAWZfWAkuyeQUytO7+7N4QKrDh+drA==" crossorigin=""/>',
        extra_css,
        "    <script src='https://unpkg.com/leaflet@1.3.4/dist/leaflet.js' integrity='sha512-nMMmRyTVoLYqjP9hrbed9S+FzjZHW5gY1TWCHA5ckwXZBadntCNs8kEqAWdrb9O7rxbCaA4lKTIWjDXZxflOcA==' crossorigin=''></script>",
        extra_js,
        "    <style>",
        "        html, body {",
        "            height: 100%;",
        "            margin: 0;",
        "        }",
        "        #map {",
        "            width: 100%;",
        "            height: 100%;",
        "        }",
        "    </style>",
        "</head>",
        "<body>",
        '    <div id="map"></div>',
        '    <script src="js/urlCoords.js"></script>',
        '    <script src="js/index.js"></script>',
        "</body>",
        f"<!--Made with fitsmap v{utils.get_version()}-->",
        "</html>\n",
    ]

    return "\n".join(html)
