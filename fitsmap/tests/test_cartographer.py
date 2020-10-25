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
"""Tests cartographer.py"""

import filecmp
import os

import pytest

import fitsmap.cartographer as c
import fitsmap.tests.helpers as helpers


@pytest.mark.unit
def test_layer_name_to_dict():
    """test cartographer.layer_name_to_dict"""
    min_zoom = 0
    max_zoom = 2
    name = "test"

    actual_dict = c.layer_name_to_dict(min_zoom, max_zoom, name)

    expected_dict = dict(
        directory=name + "/{z}/{y}/{x}.png",
        name=name,
        min_zoom=min_zoom,
        max_zoom=max_zoom + 5,
        max_native_zoom=max_zoom,
    )

    assert expected_dict == actual_dict


@pytest.mark.unit
def test_layer_dict_to_str():
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

    actual_str = c.layer_dict_to_str(layer_dict)

    expected_str = "".join(
        [
            "   var " + layer_dict["name"],
            ' = L.tileLayer("' + layer_dict["directory"] + '"',
            ", { ",
            'attribution:"'
            + "<a href='https://github.com/ryanhausen/fitsmap'>FitsMap</a>"
            + '",',
            "minZoom: " + str(layer_dict["min_zoom"]) + ",",
            "maxZoom: " + str(layer_dict["max_zoom"]) + ",",
            "maxNativeZoom: " + str(layer_dict["max_native_zoom"]) + ",",
            "}).addTo(map);",
        ]
    )

    assert expected_str == actual_str


@pytest.mark.unit
def test_layers_dict_to_base_layer_js():
    """Test cartographer.layers_dict_to_base_layer_js"""

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

    acutal_base_layers = c.layers_dict_to_base_layer_js([layer_dict])

    expected_base_layers = "\n".join(
        ["   var baseLayers = {", '      "{0}": {0},'.format(name), "   };",]
    )

    assert expected_base_layers == acutal_base_layers


@pytest.mark.unit
def test_layer_names_to_layer_control_full():
    """test cartographer.layers_dict_to_layer_control with entries"""

    actual_layer_control = c.layer_names_to_layer_control(["test"])

    expected_layer_control = "\n".join(
        [
            "   var overlays = {}",
            "",
            "   for(i = 0; i < markers.length; i++) {",
            "      overlays[labels[i]] = markers[i];",
            "   }",
            "",
            "   var layerControl = L.control.layers(baseLayers, overlays);",
            "   layerControl.addTo(map);",
        ]
    )

    assert expected_layer_control == actual_layer_control


@pytest.mark.unit
def test_layer_names_to_layer_control_empty():
    """test cartographer.layers_dict_to_layer_control without entries"""

    actual_layer_control = c.layer_names_to_layer_control([])

    expected_layer_control = "   L.control.layers(baseLayers, {}).addTo(map);"

    assert expected_layer_control == actual_layer_control


@pytest.mark.unit
def test_colors_js():
    """test cartographer.colors_js"""
    expected = "\n".join(
        [
            "   let colors = [",
            '      "#4C72B0",',
            '      "#DD8452",',
            '      "#55A868",',
            '      "#C44E52",',
            '      "#8172B3",',
            '      "#937860",',
            '      "#DA8BC3",',
            '      "#8C8C8C",',
            '      "#CCB974",',
            '      "#64B5CD",',
            "   ];",
        ]
    )

    assert expected == c.colors_js()


@pytest.mark.unit
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
            '   var map = L.map("map", {',
            "      crs: L.CRS.FitsMap,",
            "      zoom: " + str(min_zoom) + ",",
            "      minZoom: " + str(min_zoom) + ",",
            "      center:[-126, 126],",
            "   });",
        ]
    )

    assert expected_map_js == acutal_map_js


@pytest.mark.unit
def test_markers_to_js():
    """test cartographer.markers_to_js"""

    name = "test.cat.js"

    actual_marker_js = c.markers_to_js([name])

    expected_marker_js = "\n".join(
        [
            "   var markers = [",
            "      L.markerClusterGroup({ }),",
            "   ];",
            "",
            "   var markerList = [",
            "      [],",
            "   ];",
            "",
            "   var collections = [",
            "      test_cat_var,",
            "   ];",
            "",
            "   var labels = [",
            "      'test',",
            "   ];",
            "",
            "   let colors = [",
            '      "#4C72B0",',
            '      "#DD8452",',
            '      "#55A868",',
            '      "#C44E52",',
            '      "#8172B3",',
            '      "#937860",',
            '      "#DA8BC3",',
            '      "#8C8C8C",',
            '      "#CCB974",',
            '      "#64B5CD",',
            "   ];",
            "",
            "   for (i = 0; i < collections.length; i++){",
            "      collection = collections[i];",
            "",
            "      for (j = 0; j < collection.length; j++){",
            "         src = collection[j];",
            "",
            "         markerList[i].push(L.circleMarker([src.y, src.x], {",
            "            catalog_id: labels[i] + ':' + src.catalog_id + ':',",
            "            color: colors[i % colors.length]",
            "         }).bindPopup(src.desc))",
            "      }",
            "   }",
            "",
            "   for (i = 0; i < collections.length; i++){",
            "      markers[i].addLayers(markerList[i]);",
            "   }",
        ]
    )

    assert expected_marker_js == actual_marker_js


@pytest.mark.unit
def test_build_conditional_css():
    """test cartographer.build_conditional_css"""

    helpers.setup()

    actual_css = c.build_conditional_css(helpers.TEST_PATH)

    expected_css = "\n".join(
        [
            "   <link rel='stylesheet' href='https://unpkg.com/leaflet-search@2.9.8/dist/leaflet-search.src.css'/>",
            "   <link rel='stylesheet' href='css/MarkerCluster.Default.css'/>",
            "   <link rel='stylesheet' href='css/MarkerCluster.css'/>",
        ]
    )

    helpers.tear_down()

    assert expected_css == actual_css


@pytest.mark.unit
def test_build_conditional_js():
    """test cartographer.build_conditional_js"""

    test_file = "test.js"

    acutal_js = c.build_conditional_js([test_file])

    expected_js = "\n".join(
        [
            "   <script src='https://unpkg.com/leaflet.markercluster@1.4.1/dist/leaflet.markercluster-src.js' crossorigin=''></script>",
            "   <script src='https://unpkg.com/leaflet-search@2.9.8/dist/leaflet-search.src.js' crossorigin=''></script>",
            "   <script src='js/test.js'></script>",
        ]
    )

    assert expected_js == acutal_js


@pytest.mark.unit
def test_build_html():
    """test cartographer.build_html"""

    title = "test_title"
    js = "test_js"
    extra_js = "test_extra_js"
    extra_css = "test_extra_css"

    actual_html = c.build_html(title, js, extra_js, extra_css)

    expected_html = "\n".join(
        [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "   <title>{}</title>".format(title),
            '   <meta charset="utf-8" />',
            '   <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            '   <link rel="shortcut icon" type="image/x-icon" href="docs/images/favicon.ico" />',
            '   <link rel="stylesheet" href="https://unpkg.com/leaflet@1.3.4/dist/leaflet.css" integrity="sha512-puBpdR0798OZvTTbP4A8Ix/l+A4dHDD0DGqYW6RQ+9jxkRFclaxxQb/SJAWZfWAkuyeQUytO7+7N4QKrDh+drA==" crossorigin=""/>',
            extra_css,
            "   <script src='https://unpkg.com/leaflet@1.3.4/dist/leaflet.js' integrity='sha512-nMMmRyTVoLYqjP9hrbed9S+FzjZHW5gY1TWCHA5ckwXZBadntCNs8kEqAWdrb9O7rxbCaA4lKTIWjDXZxflOcA==' crossorigin=''></script>",
            extra_js,
            "   <style>",
            "       html, body {",
            "       height: 100%;",
            "       margin: 0;",
            "       }",
            "       #map {",
            "           width: 100%;",
            "           height: 100%;",
            "       }",
            "   </style>",
            "</head>",
            "<body>",
            '   <div id="map"></div>',
            "   <script>",
            js,
            "   </script>",
            "</body>",
            "</html>",
        ]
    )

    assert expected_html == actual_html


@pytest.mark.integration
def test_chart():
    """test cartographer.chart"""

    helpers.setup(with_data=True)

    out_dir = helpers.TEST_PATH
    title = "test"
    map_layer_names = "test_layer"
    marker_file_names = "test_marker"

    list(
        map(
            lambda r: os.makedirs(os.path.join(out_dir, map_layer_names, str(r))),
            range(2),
        )
    )

    c.chart(out_dir, title, [map_layer_names], [marker_file_names])

    expected_html = os.path.join(out_dir, "test_index.html")
    actual_html = os.path.join(out_dir, "index.html")

    files_match = filecmp.cmp(expected_html, actual_html)

    helpers.tear_down()

    assert files_match
