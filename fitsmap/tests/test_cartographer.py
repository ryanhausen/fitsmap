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

import pytest

import fitsmap.cartographer as c


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
            "});",
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

    expected_map_js = "\n".join([
        '   var map = L.map("map", {',
        "      crs: L.CRS.Simple,",
        "      zoom: " + str(min_zoom) + ",",
        "      minZoom: " + str(min_zoom) + ",",
        "      center:[-126, 126],",
        "      layers:[{}]".format(name),
        "   });",
    ])

    assert expected_map_js == acutal_map_js