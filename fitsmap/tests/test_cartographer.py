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
"""Tests cartographer.py"""

import filecmp
import os

import numpy as np
from astropy.wcs.wcs import WCS

import pytest

import fitsmap.cartographer as c
import fitsmap.tests.helpers as helpers


@pytest.mark.unit
@pytest.mark.cartographer
def test_layer_name_to_dict_image():
    """test cartographer.layer_name_to_dict"""
    out_dir = "."
    min_zoom = 0
    max_native_zoom = 2
    name = "test"
    color = ""  # "#4C72B0"

    actual_dict = c.layer_name_to_dict(
        out_dir, max_native_zoom + 5, min_zoom, max_native_zoom, name, color
    )

    expected_dict = dict(
        directory=name + "/{z}/{y}/{x}.png",
        name=name,
        min_zoom=min_zoom,
        max_zoom=max_native_zoom + 5,
        max_native_zoom=max_native_zoom,
    )

    assert expected_dict == actual_dict


@pytest.mark.unit
@pytest.mark.cartographer
def test_layer_name_to_dict_catalog():
    """test cartographer.layer_name_to_dict"""
    helpers.setup()
    out_dir = helpers.DATA_DIR
    min_zoom = 0
    max_native_zoom = 2
    name = "test"
    color = "#4C72B0"
    columns = "a,b,c"

    with open(os.path.join(out_dir, f"{name}.columns"), "w") as f:
        f.write(columns)

    actual_dict = c.layer_name_to_dict(
        out_dir, max_native_zoom + 5, min_zoom, max_native_zoom, name, color
    )

    expected_dict = dict(
        directory=name + "/{z}/{y}/{x}.pbf",
        name=name,
        min_zoom=min_zoom,
        max_zoom=max_native_zoom + 5,
        max_native_zoom=max_native_zoom,
        stroke_color=color,
        fill_color=color,
        stroke_opacity=1.0,
        fill_opacity=0.2,
        columns=[f'"{c}"' for c in columns.split(",")],
    )

    helpers.tear_down()

    assert expected_dict == actual_dict


@pytest.mark.unit
@pytest.mark.cartographer
def test_img_layer_dict_to_str():
    """test cartographer.layer_dict_to_str"""

    min_zoom = 0
    max_zoom = 2
    name = "test"

    layer_dict = dict(
        directory=name + "/{z}/{y}/{x}.png",
        name=name,
        min_zoom=min_zoom,
        max_zoom=max_zoom + 5,
        max_native_zoom=max_zoom,
    )

    actual_str = c.img_layer_dict_to_str(layer_dict)

    expected_str = "".join(
        [
            "const " + layer_dict["name"],
            ' = L.tileLayer("' + layer_dict["directory"] + '"',
            ", { ",
            'attribution:"'
            + "<a href='https://github.com/ryanhausen/fitsmap'>FitsMap</a>"
            + '", ',
            "minZoom: " + str(layer_dict["min_zoom"]) + ", ",
            "maxZoom: " + str(layer_dict["max_zoom"]) + ", ",
            "maxNativeZoom: " + str(layer_dict["max_native_zoom"]) + " ",
            "});",
        ]
    )

    assert expected_str == actual_str


@pytest.mark.unit
@pytest.mark.cartographer
def test_cat_layer_dict_to_str():
    """test cartographer.layer_dict_to_str"""

    min_zoom = 0
    max_zoom = 2
    name = "test"
    color = "red"
    columns = "a,b,c"

    layer_dict = dict(
        directory=name + "/{z}/{y}/{x}.png",
        name=name,
        min_zoom=min_zoom,
        max_zoom=max_zoom + 5,
        max_native_zoom=max_zoom,
        stroke_color="red",
        fill_color="red",
        columns=[f'"{c}"' for c in columns.split(",")],
    )

    actual_str = c.cat_layer_dict_to_str(layer_dict, "Infinity")

    expected_str = "".join(
        [
            "const " + layer_dict["name"],
            " = L.gridLayer.tiledMarkers(",
            "{ ",
            'tileURL:"' + layer_dict["directory"] + '", ',
            "radius: 10, ",
            'strokeColor: "' + layer_dict["stroke_color"] + '", ',
            'fillColor: "' + layer_dict["fill_color"] + '", ',
            "fillOpacity: 0.2, ",
            "strokeOpacity: 1.0, ",
            f"nCols: Infinity, ",
            f'catalogColumns: [{",".join(layer_dict["columns"])}], ',
            "minZoom: " + str(layer_dict["min_zoom"]) + ", ",
            "maxZoom: " + str(layer_dict["max_zoom"]) + ", ",
            "maxNativeZoom: " + str(layer_dict["max_native_zoom"]) + " ",
            "});",
        ]
    )

    helpers.tear_down()

    assert expected_str == actual_str


@pytest.mark.unit
@pytest.mark.cartographer
def test_leaflet_layer_control_declaration():
    """test cartographer.add_layer_control"""
    min_zoom = 0
    max_zoom = 2
    name = "test"

    img_layer_dict = dict(
        directory=name + "/{z}/{y}/{x}.png",
        name=name,
        min_zoom=min_zoom,
        max_zoom=max_zoom + 5,
        max_native_zoom=max_zoom,
    )

    cat_layer_dict = dict(
        directory=name + "/{z}/{y}/{x}.pbf",
        name=name,
        min_zoom=min_zoom,
        max_zoom=max_zoom + 5,
        max_native_zoom=max_zoom,
        color="red",
    )

    actual = c.leaflet_layer_control_declaration([img_layer_dict], [cat_layer_dict])

    expected = "\n".join(
        [
            "const catalogs = {",
            '    "test":test',
            "};",
            "",
            "const layerControl = L.control.layers(",
            '    {"test":test},',
            "    catalogs",
            ").addTo(map);",
        ]
    )

    assert expected == actual


@pytest.mark.unit
@pytest.mark.cartographer
def test_get_colors():
    """test cartographer.colors_js"""
    expected = [
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

    color_iter = c.get_colors()

    assert expected == [next(color_iter) for _ in range(len(expected))]


@pytest.mark.unit
@pytest.mark.cartographer
def test_leaflet_crs_js():
    """test cartographer.leaflet_crs_js"""
    min_zoom = 0
    max_zoom = 2
    name = "test"

    layer_dict = dict(
        directory=name + "/{z}/{y}/{x}.png",
        name=name,
        min_zoom=min_zoom,
        max_zoom=max_zoom + 5,
        max_native_zoom=max_zoom,
    )

    actual = c.leaflet_crs_js([layer_dict])

    expected = "\n".join(
        [
            "L.CRS.FitsMap = L.extend({}, L.CRS.Simple, {",
            f"    transformation: new L.Transformation(1/{int(2**max_zoom)}, 0, -1/{int(2**max_zoom)}, 256)",
            "});",
        ]
    )

    assert actual == expected


@pytest.mark.unit
@pytest.mark.cartographer
def test_extract_cd_matrix_as_string_with_cd():
    """test cartographer.extract_cd_matrix_as_string"""

    wcs = helpers.MockWCS(include_cd=True)

    actual = c.extract_cd_matrix_as_string(wcs)

    expected = "[[1, 0], [0, 1]]"

    assert actual == expected


@pytest.mark.unit
@pytest.mark.cartographer
def test_extract_cd_matrix_as_string_without_cd():
    """test cartographer.extract_cd_matrix_as_string"""

    wcs = helpers.MockWCS(include_cd=False)

    actual = c.extract_cd_matrix_as_string(wcs)

    expected = "[[0.0, 0.0], [0.0, 0.0]]"

    assert actual == expected


@pytest.mark.unit
@pytest.mark.cartographer
def test_leaflet_map_js():
    """test cartographer.leaflet_map_js"""

    min_zoom = 0
    max_zoom = 2
    name = "test"

    layer_dict = dict(
        directory=name + "/{z}/{y}/{x}.png",
        name=name,
        min_zoom=min_zoom,
        max_zoom=max_zoom + 5,
        max_native_zoom=max_zoom,
    )

    acutal_map_js = c.leaflet_map_js([layer_dict])

    expected_map_js = "\n".join(
        [
            'const map = L.map("map", {',
            "    crs: L.CRS.FitsMap,",
            "    minZoom: " + str(min_zoom) + ",",
            "    preferCanvas: true,",
            f"    layers: [{layer_dict['name']}]",
            "});",
        ]
    )

    assert expected_map_js == acutal_map_js


@pytest.mark.unit
@pytest.mark.cartographer
def test_build_conditional_css():
    """test cartographer.build_conditional_css"""

    helpers.setup()

    actual_css = c.build_conditional_css(helpers.TEST_PATH)

    expected_css = "\n".join(
        [
            "    <link rel='preload' href='https://unpkg.com/leaflet-search@2.9.8/dist/leaflet-search.src.css'  as='style' onload='this.rel=\"stylesheet\"'/>",
            "    <link rel='preload' href='css/LabelControl.min.css'  as='style' onload='this.rel=\"stylesheet\"'/>",
            "    <link rel='preload' href='css/MarkerCluster.Default.min.css'  as='style' onload='this.rel=\"stylesheet\"'/>",
            "    <link rel='preload' href='css/MarkerCluster.min.css'  as='style' onload='this.rel=\"stylesheet\"'/>",
            "    <link rel='preload' href='css/MarkerPopup.min.css'  as='style' onload='this.rel=\"stylesheet\"'/>",
            "    <link rel='preload' href='css/SettingsControl.min.css'  as='style' onload='this.rel=\"stylesheet\"'/>",
            "    <link rel='preload' href='css/TileNearestNeighbor.min.css'  as='style' onload='this.rel=\"stylesheet\"'/>",
        ]
    )

    helpers.tear_down()

    assert expected_css == actual_css


@pytest.mark.unit
@pytest.mark.cartographer
def test_build_conditional_js():
    """test cartographer.build_conditional_js"""

    helpers.setup()

    acutal_js = c.build_conditional_js(helpers.TEST_PATH, True)

    expected_js = "\n".join(
        [
            "    <script defer src='https://cdnjs.cloudflare.com/ajax/libs/leaflet-search/3.0.2/leaflet-search.src.min.js'></script>",
            "    <script defer src='js/customSearch.min.js'></script>",
            "    <script defer src='js/tiledMarkers.min.js'></script>",
            "    <script defer src='https://cdn.jsdelivr.net/npm/toolcool-color-picker/dist/toolcool-color-picker.min.js'></script>",
            "    <script defer src='js/fitsmapScale.min.js'></script>",
            "    <script defer src='js/labelControl.min.js'></script>",
            "    <script defer src='js/settingsControl.min.js'></script>",
            "    <script defer src='js/urlCoords.js'></script>",
            "    <script defer src='js/index.js'></script>",
            "    <script defer src='https://unpkg.com/cbor-web@8.1.0/dist/cbor.js'></script>",
            "    <script defer src='https://unpkg.com/pbf@3.0.5/dist/pbf.js'></script>",
            "    <script defer src='js/l.ellipse.min.js'></script>",
            "    <script defer src='js/vector-tile.min.js'></script>",
        ]
    )

    helpers.tear_down()

    assert expected_js == acutal_js


@pytest.mark.unit
@pytest.mark.cartographer
def test_build_index_js():
    """Tests cartographer.build_index_js"""

    img_layer_dict = [
        dict(
            name="img",
            directory="img/{z}/{y}/{x}.png",
            min_zoom=0,
            max_zoom=8,
            max_native_zoom=3,
        )
    ]

    cat_layer_dict = [
        dict(
            name="cat",
            directory="cat/{z}/{y}/{x}.pbf",
            min_zoom=0,
            max_zoom=8,
            max_native_zoom=2,
            stroke_color="rgb(76, 114, 176)",
            fill_color="rgb(76, 114, 176)",
            columns=['"a"', '"b"', '"c"'],
        )
    ]

    n_cols = 1
    max_xy = (2048, 2048)
    pixel_scale = 0.06
    units_are_pixels = True

    expected_js = "\n".join(
        [
            "// Image layers ================================================================",
            'const img = L.tileLayer("img/{z}/{y}/{x}.png", { attribution:"'
            + c.LAYER_ATTRIBUTION
            + '", '
            + "minZoom: 0, maxZoom: 8, maxNativeZoom: 3 });",
            "",
            "// Marker layers ===============================================================",
            'const cat = L.gridLayer.tiledMarkers({ tileURL:"cat/{z}/{y}/{x}.pbf", radius: 10, strokeColor: "rgb(76, 114, 176)", fillColor: "rgb(76, 114, 176)", fillOpacity: 0.2, strokeOpacity: 1.0, nCols: 1, catalogColumns: ["a","b","c"], minZoom: 0, maxZoom: 8, maxNativeZoom: 2 });',
            "",
            "// Basic map setup =============================================================",
            "L.CRS.FitsMap = L.extend({}, L.CRS.Simple, {",
            "    transformation: new L.Transformation(1/8, 0, -1/8, 256)",
            "});",
            "",
            'const map = L.map("map", {',
            "    crs: L.CRS.FitsMap,",
            "    minZoom: 0,",
            "    preferCanvas: true,",
            "    layers: [img]",
            "});",
            "",
            "// Scale Bar Control ===========================================================",
            "// https://stackoverflow.com/a/62093918",
            "const scale = L.control.fitsmapScale({",
            f"    pixelScale: {pixel_scale},",
            f"    unitsArePixels: true,",
            "}).addTo(map);",
            "",
            "// Label Control ===============================================================",
            "const label = L.control.label({",
            "    position: 'bottomleft',",
            "    title: '',",
            "    isRADec: Boolean(is_ra_dec) // from urlCoords.js",
            "}).addTo(map);",
            "",
            "const catalogs = {",
            '    "cat":cat',
            "};",
            "",
            "const layerControl = L.control.layers(",
            '    {"img":img},',
            "    catalogs",
            ").addTo(map);",
            "",
            "// Search ======================================================================",
            "const catalogPaths = [",
            '    "catalog_assets/cat/",',
            "];",
            "",
            "const searchControl = buildCustomSearch(catalogPaths, 2);",
            "map.addControl(searchControl);",
            "",
            "// Settings Control ============================================================",
            "const settingsControl = L.control.settings({",
            "    position: 'topleft',",
            "    catalogs:catalogs,",
            "}).addTo(map);",
            "",
            "// Map event setup =============================================================",
            'img.on("load", () => {',
            '    document.getElementById("loading-screen").style.display = "none";',
            '    document.getElementById("map").style.visibility = "visible";',
            "    label.update(map.getCenter());",
            "});",
            "",
            'map.on("moveend", updateLocationBar);',
            'map.on("zoomend", updateLocationBar);',
            'map.on("mousemove", (event) => {label.update(event.latlng);});',
            'map.on("baselayerchange", (event) => {label.options.title = event.name;});',
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

    actual_js = c.build_index_js(
        img_layer_dict, cat_layer_dict, n_cols, max_xy, pixel_scale, units_are_pixels
    )

    assert expected_js == actual_js


@pytest.mark.unit
@pytest.mark.cartographer
def test_move_support_images():
    """test cartographer.move_support_images"""

    helpers.setup()

    actual_moved_images = c.move_support_images(helpers.TEST_PATH)

    expected_moved_images = ["favicon.ico", "loading-logo.svg"]

    helpers.tear_down()

    assert actual_moved_images == expected_moved_images


@pytest.mark.unit
@pytest.mark.cartographer
def test_build_html():
    """test cartographer.build_html"""

    title = "test_title"
    extra_js = "test_extra_js"
    extra_css = "test_extra_css"

    actual_html = c.build_html(title, extra_js, extra_css)

    expected_html = "\n".join(
        [
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
            '    Copyright (c) 2023 by kootoopas (https://codepen.io/kootoopas/pen/kGPoaB) Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.',
            "    */",
            '    @-webkit-keyframes bg-scrolling-reverse {100% {background-position: 50px 50px;}}@-moz-keyframes bg-scrolling-reverse {100% {background-position: 50px 50px;}}@-o-keyframes bg-scrolling-reverse {100% {background-position: 50px 50px;}}@keyframes bg-scrolling-reverse {100% {background-position: 50px 50px;}}@-webkit-keyframes bg-scrolling {0% {background-position: 50px 50px;}}@-moz-keyframes bg-scrolling {0% {background-position: 50px 50px;}}@-o-keyframes bg-scrolling {0% {background-position: 50px 50px;}}@keyframes bg-scrolling {0% {background-position: 50px 50px;}}#loading-screen {color: #999;font: 400 16px/1.5 exo, ubuntu, "segoe ui", helvetica, arial, sans-serif;text-align: center;background: url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAyCAIAAACRXR/mAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAIGNIUk0AAHolAACAgwAA+f8AAIDpAAB1MAAA6mAAADqYAAAXb5JfxUYAAABnSURBVHja7M5RDYAwDEXRDgmvEocnlrQS2SwUFST9uEfBGWs9c97nbGtDcquqiKhOImLs/UpuzVzWEi1atGjRokWLFi1atGjRokWLFi1atGjRokWLFi1af7Ukz8xWp8z8AAAA//8DAJ4LoEAAlL1nAAAAAElFTkSuQmCC") repeat 0 0;-webkit-animation: bg-scrolling-reverse 0.92s infinite;-moz-animation: bg-scrolling-reverse 0.92s infinite;-o-animation: bg-scrolling-reverse 0.92s infinite;animation: bg-scrolling-reverse 0.92s infinite;-webkit-animation-timing-function: linear;-moz-animation-timing-function: linear;-o-animation-timing-function: linear;animation-timing-function: linear;width: 100%;height: 100%;}',
            "    </style>",
            "</head>",
            "<body>",
            '    <div id="loading-screen" class="overlay">',
            '        <div class="brand"><img src="imgs/loading-logo.svg" style="width: 100%" alt="FitsMap logo"/></div>',
            '        <div class="loading"></div>',
            '        <div class="loadingtext">Loading...</div>',
            "    </div>",
            '    <div id="map"></div>',
            "</body>",
            f"<!--Made with fitsmap v{helpers.get_version()}-->",
            "</html>\n",
        ]
    )

    assert expected_html == actual_html


@pytest.mark.integration
@pytest.mark.cartographer
def test_chart_no_wcs():
    """test cartographer.chart"""

    helpers.setup(with_data=True)

    out_dir = helpers.TEST_PATH
    title = "test"
    map_layer_names = "test_layer"
    marker_file_names = "test_marker"
    wcs = None
    columns = "a,b,c"
    pixel_scale = 0.5
    units_are_pixels = True

    with open(os.path.join(out_dir, f"{marker_file_names}.columns"), "w") as f:
        f.write(columns)

    list(
        map(
            lambda r: os.makedirs(os.path.join(out_dir, map_layer_names, str(r))),
            range(3),
        )
    )

    # make mock marker zooms
    list(
        map(
            lambda r: os.makedirs(os.path.join(out_dir, marker_file_names, str(r))),
            range(2),
        )
    )

    os.mkdir(os.path.join(out_dir, "js"))
    os.mkdir(os.path.join(out_dir, "css"))

    c.chart(
        out_dir,
        title,
        [map_layer_names],
        [marker_file_names],
        wcs,
        float("inf"),
        [100, 100],
        pixel_scale,
        units_are_pixels,
    )

    # inject current version in to test_index.html
    version = helpers.get_version()
    raw_path = os.path.join(out_dir, "test_index.html")
    with open(raw_path, "r") as f:
        converted = list(map(lambda l: l.replace("VERSION", version), f.readlines()))

    with open(raw_path, "w") as f:
        f.writelines(converted)

    actual_html = os.path.join(out_dir, "index.html")
    expected_html = os.path.join(out_dir, "test_index.html")

    files_match = filecmp.cmp(expected_html, actual_html)

    helpers.tear_down()

    assert files_match


@pytest.mark.integration
@pytest.mark.cartographer
def test_chart_with_wcs():
    """test cartographer.chart"""

    helpers.setup(with_data=True)

    out_dir = helpers.TEST_PATH
    title = "test"
    map_layer_names = "test_layer"
    marker_file_names = "test_marker"
    wcs = WCS(os.path.join(out_dir, "test_image.fits"))
    columns = "a,b,c"
    pixel_scale = 0.5
    units_are_pixels = True

    with open(os.path.join(out_dir, f"{marker_file_names}.columns"), "w") as f:
        f.write(columns)

    # make mock image zooms
    list(
        map(
            lambda r: os.makedirs(os.path.join(out_dir, map_layer_names, str(r))),
            range(3),
        )
    )

    # make mock marker zooms
    list(
        map(
            lambda r: os.makedirs(os.path.join(out_dir, marker_file_names, str(r))),
            range(2),
        )
    )

    os.mkdir(os.path.join(out_dir, "js"))
    os.mkdir(os.path.join(out_dir, "css"))

    c.chart(
        out_dir,
        title,
        [map_layer_names],
        [marker_file_names],
        wcs,
        float("inf"),
        [100, 100],
        pixel_scale,
        units_are_pixels,
    )

    # inject current version in to test_index.html
    version = helpers.get_version()
    raw_path = os.path.join(out_dir, "test_index_wcs.html")
    with open(raw_path, "r") as f:
        converted = list(map(lambda l: l.replace("VERSION", version), f.readlines()))

    with open(raw_path, "w") as f:
        f.writelines(converted)

    actual_html = os.path.join(out_dir, "index.html")
    expected_html = os.path.join(out_dir, "test_index_wcs.html")

    files_match = filecmp.cmp(expected_html, actual_html)

    helpers.tear_down()

    assert files_match
