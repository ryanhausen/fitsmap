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
"""Tests cartographer.py"""

import filecmp
import os
from astropy.wcs.wcs import WCS

import pytest

import fitsmap.cartographer as c
import fitsmap.tests.helpers as helpers


@pytest.mark.unit
@pytest.mark.cartographer
def test_layer_name_to_dict_image():
    """test cartographer.layer_name_to_dict"""
    min_zoom = 0
    max_zoom = 2
    name = "test"
    color = ""  # "#4C72B0"

    actual_dict = c.layer_name_to_dict(min_zoom, max_zoom, name, color)

    expected_dict = dict(
        directory=name + "/{z}/{y}/{x}.png",
        name=name,
        min_zoom=min_zoom,
        max_zoom=max_zoom + 5,
        max_native_zoom=max_zoom,
    )

    assert expected_dict == actual_dict


@pytest.mark.unit
@pytest.mark.cartographer
def test_layer_name_to_dict_catalog():
    """test cartographer.layer_name_to_dict"""
    min_zoom = 0
    max_zoom = 2
    name = "test"
    color = "#4C72B0"

    actual_dict = c.layer_name_to_dict(min_zoom, max_zoom, name, color)

    expected_dict = dict(
        directory=name + "/{z}/{y}/{x}.pbf",
        name=name,
        min_zoom=min_zoom,
        max_zoom=max_zoom + 5,
        max_native_zoom=max_zoom,
        color=color,
    )

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

    layer_dict = dict(
        directory=name + "/{z}/{y}/{x}.png",
        name=name,
        min_zoom=min_zoom,
        max_zoom=max_zoom + 5,
        max_native_zoom=max_zoom,
        color="red",
    )

    actual_str = c.cat_layer_dict_to_str(layer_dict, float("inf"))

    expected_str = "".join(
        [
            "const " + layer_dict["name"],
            " = L.gridLayer.tiledMarkers(",
            "{ ",
            'tileURL:"' + layer_dict["directory"] + '", ',
            'color: "' + layer_dict["color"] + '", ',
            f"rowsPerColumn: Infinity, ",
            "minZoom: " + str(layer_dict["min_zoom"]) + ", ",
            "maxZoom: " + str(layer_dict["max_zoom"]) + ", ",
            "maxNativeZoom: " + str(layer_dict["max_native_zoom"]) + " ",
            "});",
        ]
    )

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
            "const layerControl = L.control.layers(",
            '    {"test":test},',
            '    {"test":test}',
            ").addTo(map);",
        ]
    )

    assert expected == actual


@pytest.mark.unit
@pytest.mark.cartographer
def test_get_colors():
    """test cartographer.colors_js"""
    expected = [
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

    assert expected == c.get_colors()


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
            "    <link rel='stylesheet' href='https://unpkg.com/leaflet-search@2.9.8/dist/leaflet-search.src.css'/>",
            "    <link rel='stylesheet' href='css/MarkerCluster.Default.css'/>",
            "    <link rel='stylesheet' href='css/MarkerCluster.css'/>",
            "    <link rel='stylesheet' href='css/MarkerPopup.css'/>",
            "    <link rel='stylesheet' href='css/TileNearestNeighbor.css'/>",
        ]
    )

    helpers.tear_down()

    assert expected_css == actual_css


@pytest.mark.unit
@pytest.mark.cartographer
def test_build_conditional_js():
    """test cartographer.build_conditional_js"""

    helpers.setup()

    acutal_js = c.build_conditional_js(helpers.TEST_PATH)

    expected_js = "\n".join(
        [
            '    <script src="https://unpkg.com/pbf@3.0.5/dist/pbf.js", crossorigin=""></script>',
            '    <script src="https://cdn.jsdelivr.net/npm/leaflet-search" crossorigin=""></script>',
            "    <script src='js/customSearch.min.js'></script>",
            "    <script src='js/l.ellipse.min.js'></script>",
            "    <script src='js/tiledMarkers.min.js'></script>",
            "    <script src='js/vector-tile.min.js'></script>",
        ]
    )

    helpers.tear_down()

    assert expected_js == acutal_js


@pytest.mark.unit
@pytest.mark.cartographer
def test_move_support_images():
    """test cartographer.move_support_images"""

    helpers.setup()

    actual_moved_images = c.move_support_images(helpers.TEST_PATH)

    expected_moved_images = ["favicon.ico"]

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

    list(
        map(
            lambda r: os.makedirs(os.path.join(out_dir, map_layer_names, str(r))),
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

    list(
        map(
            lambda r: os.makedirs(os.path.join(out_dir, map_layer_names, str(r))),
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
