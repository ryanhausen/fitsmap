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

"""Helper functions for creating a leaflet JS HTML map."""

# ******************************************************************************
# Designed for internal use. Any method/variable can be deprecated/changed
# without consideration.
# ******************************************************************************

import os
import shutil
from itertools import count, cycle, repeat, starmap
from functools import partial, reduce
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

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
    n_cols: int,
    max_xy: Tuple[int, int],
    pixel_scale: float,
    units_are_pixels: bool,
) -> None:
    """Creates an HTML file containing a leaflet js map using the given params.

    ****************************************************************************
    * Designed for internal use. Any method/variable can be deprecated/changed *
    * without consideration.                                                   *
    ****************************************************************************
    """
    # convert layer names into a single javascript string
    layer_zooms = lambda l: list(map(int, os.listdir(os.path.join(out_dir, l))))
    img_zooms = reduce(lambda x, y: x + y, list(map(layer_zooms, img_layer_names)), [0])
    cat_zooms = reduce(
        lambda x, y: x + y, list(map(layer_zooms, marker_layer_names)), [0]
    )
    # be able to zoom in 5 levels further than the native zoom
    # this seems to work well in general, but could become a parameter.
    max_overall_zoom = max(img_zooms + cat_zooms) + 5

    convert_layer_name_func = partial(layer_name_to_dict, out_dir, max_overall_zoom)
    img_layer_dicts = list(
        starmap(
            convert_layer_name_func,
            zip(
                repeat(min(img_zooms)),
                repeat(max(img_zooms)),
                img_layer_names,
                repeat(None),
            ),
        )
    )

    cat_layer_dicts = list(
        starmap(
            convert_layer_name_func,
            zip(
                repeat(min(cat_zooms)),
                repeat(max(cat_zooms)),
                marker_layer_names,
                get_colors(),
            ),
        )
    )

    # generated javascript =====================================================
    with open(os.path.join(out_dir, "js", "urlCoords.js"), "w") as f:
        f.write(build_urlCoords_js(wcs))

    with open(os.path.join(out_dir, "js", "index.js"), "w") as f:
        f.write(
            build_index_js(
                img_layer_dicts,
                cat_layer_dicts,
                n_cols,
                max_xy,
                pixel_scale,
                units_are_pixels,
            )
        )
    # generated javascript =====================================================

    # HTML file contents =======================================================
    extra_js = build_conditional_js(out_dir, bool(cat_layer_dicts))

    extra_css = build_conditional_css(out_dir)

    move_support_images(out_dir)

    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(build_html(title, extra_js, extra_css))
    # HTML file contents =======================================================


def layer_name_to_dict(
    out_dir: str,
    max_zoom: int,
    min_zoom: int,
    max_native_zoom: int,
    name: str,
    color: str,
) -> Dict[str, Union[str, int, List[str]]]:
    """Convert layer name to dict for conversion."""

    layer_dict = dict(
        directory=name + "/{z}/{y}/{x}." + ("pbf" if color else "png"),
        name=name,
        min_zoom=min_zoom,
        max_zoom=max_zoom,
        max_native_zoom=max_native_zoom,
    )
    if color:
        layer_dict["stroke_color"] = color
        layer_dict["fill_color"] = color
        layer_dict["stroke_opacity"] = 1.0
        layer_dict["fill_opacity"] = 0.2

        cat_col_path = "/".join([out_dir, f"{name}.columns"])
        with open(cat_col_path, "r") as f:
            columns = f.readline().strip().split(",")
            layer_dict["columns"] = [f'"{c}"' for c in columns]

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


def cat_layer_dict_to_str(layer: dict, n_cols: int) -> str:
    """Convert layer dict to layer str for including in HTML file."""

    layer_str = [
        "const " + layer["name"],
        " = L.gridLayer.tiledMarkers(",
        "{ ",
        'tileURL:"' + layer["directory"] + '", ',
        "radius: 10, ",
        'strokeColor: "' + layer["stroke_color"] + '", ',
        'fillColor: "' + layer["fill_color"] + '", ',
        "fillOpacity: 0.2, ",
        "strokeOpacity: 1.0, ",
        f"nCols: {n_cols}, ",
        f'catalogColumns: [{",".join(layer["columns"])}], ',
        "minZoom: " + str(layer["min_zoom"]) + ", ",
        "maxZoom: " + str(layer["max_zoom"]) + ", ",
        "maxNativeZoom: " + str(layer["max_native_zoom"]) + " ",
        "});",
    ]

    return "".join(layer_str)


def get_colors() -> Iterable[str]:
    return cycle(
        [
            "rgb(76, 114, 176)",
            "rgb(221, 132, 82)",
            "rgb(85, 168, 104)",
            "rgb(196, 78, 82)",
            "rgb(129, 114, 179)",
            "rgb(147, 120, 96)",
            "rgb(218, 139, 195)",
            "rgb(140, 140, 140)",
            "rgb(204, 185, 116)",
            "rgb(100, 181, 205)",
        ]
    )


def leaflet_crs_js(tile_layers: List[dict]) -> str:
    max_zoom = max(map(lambda t: t["max_native_zoom"], tile_layers))

    scale_factor = int(2**max_zoom)

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


def loading_screen_js(tile_layers: List[dict]):
    js = "\n".join(
        [
            f'{tile_layers[0]["name"]}.on("load", () => {{',
            '    document.getElementById("loading-screen").style.visibility = "hidden";',
            '    document.getElementById("map").style.visibility = "visible";',
            "    label.update(map);",
            "});",
        ]
    )

    return js


def move_support_images(out_dir: str) -> List[str]:
    img_extensions = [".png", ".jpg", ".ico", ".svg"]

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
    css_string = "    <link rel='preload' href='{}'  as='style' onload='this.rel=\"stylesheet\"'/>"

    support_dir = os.path.join(os.path.dirname(__file__), "support")
    out_css_dir = os.path.join(out_dir, "css")

    local_css_files = list(
        filter(lambda f: f.endswith(".min.css"), sorted(os.listdir(support_dir)))
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


def build_conditional_js(out_dir: str, include_markerjs: bool) -> str:
    support_dir = os.path.join(os.path.dirname(__file__), "support")
    out_js_dir = os.path.join(out_dir, "js")

    local_js_files = list(
        sorted(filter(lambda f: f.endswith(".min.js"), os.listdir(support_dir)))
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

    # some files have to be loaded before index.js, so that index.js can
    # sucessfully run. Other files get placed after index.js so that the map
    # can load first and then those will load in the background
    pre_index_files = [
        "https://cdnjs.cloudflare.com/ajax/libs/leaflet-search/3.0.2/leaflet-search.src.min.js"
        * include_markerjs,
        "js/customSearch.min.js" * include_markerjs,
        "js/tiledMarkers.min.js" * include_markerjs,
        "https://cdn.jsdelivr.net/npm/toolcool-color-picker/dist/toolcool-color-picker.min.js"
        * include_markerjs,
        "js/labelControl.min.js",
        "js/settingsControl.min.js",
        "js/urlCoords.js",
        "js/index.js",
    ]

    post_index_files = [
        "https://unpkg.com/cbor-web@8.1.0/dist/cbor.js" * include_markerjs,
        "https://unpkg.com/pbf@3.0.5/dist/pbf.js" * include_markerjs,
        "js/l.ellipse.min.js" * include_markerjs,
        "js/vector-tile.min.js" * include_markerjs,
    ]

    js_string = "    <script defer src='{}'></script>"
    js_tags = list(
        map(
            lambda s: js_string.format(s),
            filter(lambda x: x, pre_index_files + post_index_files),
        )
    )

    return "\n".join(js_tags)


def leaflet_layer_control_declaration(
    img_layer_dicts: List[Dict],
    cat_layer_dicts: List[Dict],
) -> str:
    img_layer_label_pairs = ",".join(
        list(map(lambda l: '"{0}":{0}'.format(l["name"]), img_layer_dicts))
    )

    cat_layer_label_pairs = ",\n".join(
        list(map(lambda l: '    "{0}":{0}'.format(l["name"]), cat_layer_dicts))
    )

    control_js = [
        "const catalogs = {",
        cat_layer_label_pairs,
        "};",
        "",
        "const layerControl = L.control.layers(",
        f"    {{{img_layer_label_pairs}}},",
        "    catalogs",
        ").addTo(map);",
    ]

    return "\n".join(control_js)


def leaflet_search_control_declaration(
    cat_layer_dicts: List[Dict],
) -> str:
    search_js = [
        "const catalogPaths = [",
        *list(
            map(
                lambda s: f'    "{"/".join(["catalog_assets", s])}/",',
                map(lambda l: l["name"], cat_layer_dicts),
            )
        ),
        "];",
        "",
        f"const searchControl = buildCustomSearch(catalogPaths, {cat_layer_dicts[0]['max_native_zoom']});",
        "map.addControl(searchControl);",
    ]

    return "\n".join(search_js) if cat_layer_dicts else ""


def extract_cd_matrix_as_string(wcs: WCS) -> str:
    if hasattr(wcs.wcs, "cd"):
        return str(wcs.wcs.cd.tolist())
    else:
        # Manual "CD" matrix
        delta = wcs.all_pix2world(
            [
                wcs.wcs.crpix,
                wcs.wcs.crpix + np.array([1, 0]),
                wcs.wcs.crpix + np.array([0, 1]),
            ],
            0,
        )

        _cd = np.array([delta[1, :] - delta[0, :], delta[2, :] - delta[0, :]])
        _cd[0, :] *= np.cos(wcs.wcs.crval[1] / 180 * np.pi)
        return str(_cd.tolist())


def build_urlCoords_js(img_wcs: WCS) -> str:
    wcs_js_file = os.path.join(os.path.dirname(__file__), "support", "urlCoords.js.tmp")

    with open(wcs_js_file, "r") as f:
        wcs_js = "".join(f.readlines())

    if img_wcs:
        wcs_js = wcs_js.replace("_IS_RA_DEC", "1")
        wcs_js = wcs_js.replace("_CRPIX", str(img_wcs.wcs.crpix.tolist()))
        wcs_js = wcs_js.replace("_CRVAL", str(img_wcs.wcs.crval.tolist()))
        wcs_js = wcs_js.replace("_CD", extract_cd_matrix_as_string(img_wcs))
    else:
        wcs_js = wcs_js.replace("_IS_RA_DEC", "0")
        wcs_js = wcs_js.replace("_CRPIX", "[1, 1]")
        wcs_js = wcs_js.replace("_CRVAL", "[0, 0]")
        wcs_js = wcs_js.replace("_CD", "[[1, 0], [0, 1]]")

    return wcs_js


def build_index_js(
    image_layer_dicts: List[Dict],
    marker_layer_dicts: List[str],
    n_cols: int,
    max_xy: Tuple[int, int],
    pixel_scale: float,
    units_are_pixels: bool,
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
                    zip(marker_layer_dicts, repeat(n_cols)),
                )
            ),
            "",
            "// Basic map setup =============================================================",
            leaflet_crs_js(image_layer_dicts),
            "",
            leaflet_map_js(image_layer_dicts),
            "",
            leaflet_scale_bar_declaration(pixel_scale, units_are_pixels),
            "",
            leaflet_label_declaration(),
            "",
            leaflet_layer_control_declaration(image_layer_dicts, marker_layer_dicts),
            "",
            "// Search ======================================================================",
            leaflet_search_control_declaration(marker_layer_dicts)
            if len(marker_layer_dicts)
            else "",
            "",
            leaflet_settings_declaration(),
            "",
            "// Map event setup =============================================================",
            loading_screen_js(image_layer_dicts),
            "",
            'map.on("moveend", updateLocationBar);',
            'map.on("zoomend", updateLocationBar);',
            'map.on("move", () => {label.update(map);});',
            "",
            "map.whenReady(function () {",
            "    scale.options.maxWidth = Math.round(map.getSize().x * 0.2);",
            "    label.addTo(map);",
            "});",
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
        '<html lang="en">',
        "<head>",
        "    <title>{}</title>".format(title),
        '    <meta charset="utf-8" />',
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        '    <link rel="shortcut icon" type="image/x-icon" href="imgs/favicon.ico" />',
        '    <link rel="preload" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/leaflet.min.css" integrity="sha512-1xoFisiGdy9nvho8EgXuXvnpR5GAMSjFwp40gSRE3NwdUdIMIKuPa7bqoUhLD0O/5tPNhteAsE5XyyMi5reQVA==" crossorigin="anonymous" referrerpolicy="no-referrer" as="style" onload="this.rel=\'stylesheet\'"/>',
        extra_css,
        '    <script defer src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/leaflet.min.js" integrity="sha512-SeiQaaDh73yrb56sTW/RgVdi/mMqNeM2oBwubFHagc5BkixSpP1fvqF47mKzPGWYSSy4RwbBunrJBQ4Co8fRWA==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>',
        extra_js,
        "    <style>",
        "    /* Map */",
       r"    html,body{height:100%;padding:0;margin:0;font-family:Helvetica,Arial,sans-serif}#map{width:100%;height:100%;visibility:hidden}",
        "    /* Loading Page */",
        "    /*",
       r"    Copyright (c) 2023 by kootoopas (https://codepen.io/kootoopas/pen/kGPoaB) Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.",
        "    */",
       r'    @-webkit-keyframes bg-scrolling-reverse {100% {background-position: 50px 50px;}}@-moz-keyframes bg-scrolling-reverse {100% {background-position: 50px 50px;}}@-o-keyframes bg-scrolling-reverse {100% {background-position: 50px 50px;}}@keyframes bg-scrolling-reverse {100% {background-position: 50px 50px;}}@-webkit-keyframes bg-scrolling {0% {background-position: 50px 50px;}}@-moz-keyframes bg-scrolling {0% {background-position: 50px 50px;}}@-o-keyframes bg-scrolling {0% {background-position: 50px 50px;}}@keyframes bg-scrolling {0% {background-position: 50px 50px;}}#loading-screen {color: #999;font: 400 16px/1.5 exo, ubuntu, "segoe ui", helvetica, arial, sans-serif;text-align: center;background: url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAyCAIAAACRXR/mAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAIGNIUk0AAHolAACAgwAA+f8AAIDpAAB1MAAA6mAAADqYAAAXb5JfxUYAAABnSURBVHja7M5RDYAwDEXRDgmvEocnlrQS2SwUFST9uEfBGWs9c97nbGtDcquqiKhOImLs/UpuzVzWEi1atGjRokWLFi1atGjRokWLFi1atGjRokWLFi1af7Ukz8xWp8z8AAAA//8DAJ4LoEAAlL1nAAAAAElFTkSuQmCC") repeat 0 0;-webkit-animation: bg-scrolling-reverse 0.92s infinite;-moz-animation: bg-scrolling-reverse 0.92s infinite;-o-animation: bg-scrolling-reverse 0.92s infinite;animation: bg-scrolling-reverse 0.92s infinite;-webkit-animation-timing-function: linear;-moz-animation-timing-function: linear;-o-animation-timing-function: linear;animation-timing-function: linear;width: 100%;height: 100%;}',
        "    </style>",
        "</head>",
        "<body>",
        '    <div id="loading-screen" class="overlay">',
        '        <div class="brand"><img src="imgs/loading-logo.svg" /></div>',
        '        <div class="loading"></div>',
        '        <div class="loadingtext">Loading...</div>',
        "    </div>",
        '    <div id="map"></div>',
        "</body>",
        f"<!--Made with fitsmap v{utils.get_version()}-->",
        "</html>\n",
    ]

    return "\n".join(html)


def leaflet_scale_bar_declaration(
    pixel_scale: float,
    units_are_pixels: bool,
) -> str:
    js_bool = "true" if units_are_pixels else "false"

    return "\n".join([
        "// Scale Bar Control ===========================================================",
        "// https://stackoverflow.com/a/62093918",
        "const scale = L.control.fitsmapScale({",
        f"    pixelScale: {pixel_scale},",
        f"    unitsArePixels: {js_bool},",
        "}).addTo(map);",
    ])


def leaflet_label_declaration() -> str:
    js = "\n".join(
        [
            "// Label Control ===============================================================",
            "const label = L.control.label({",
            "    position: 'bottomleft',",
            "    title: '',",
            "    isRADec: Boolean(is_ra_dec) // from urlCoords.js",
            "}).addTo(map);",
        ]
    )

    return js


def leaflet_settings_declaration() -> str:
    js = "\n".join(
        [
            "// Settings Control ============================================================",
            "const settingsControl = L.control.settings({",
            "    position: 'topleft',",
            "    catalogs:catalogs,",
            "}).addTo(map);",
        ]
    )

    return js
