# MIT License
# Copyright 2021 Ryan Hausen

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
@pytest.mark.cartographer
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
            "    const " + layer_dict["name"],
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
@pytest.mark.cartographer
def test_add_layer_control():
    """test cartographer.add_layer_control"""
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

    actual = c.add_layer_control(layer_dict)

    expected = "\n".join(
        [
            f'    layerControl.addBaseLayer(test, "test");',
            "    layerControl.addTo(map);",
        ]
    )

    assert expected == actual


@pytest.mark.unit
@pytest.mark.cartographer
def test_js_async_layers():
    """test cartographer.js_async_layers"""

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

    actual = c.js_async_layers([layer_dict])

    expected = "\n".join(
        [
            "    asyncLayers = [",
            '        ["test", L.tileLayer("test/{z}/{y}/{x}.png",{ attribution:"'
            + c.LAYER_ATTRIBUTION
            + '", '
            + "minZoom:0, maxZoom:7, "
            + "maxNativeZoom:2})],",
            "    ];",
        ]
    )

    assert expected == actual


@pytest.mark.unit
@pytest.mark.cartographer
def test_js_load_next_layer():
    """test cartographer.js_load_next_layer"""

    actual = c.js_load_next_layer()

    expected = "\n".join(
        [
            "    function loadNextLayer(event) {",
            "        if (asyncLayers.length > 0){",
            "            nextLayer = asyncLayers.pop()",
            '            nextLayer[1].on("load", loadNextLayer);',
            "            layerControl.addBaseLayer(nextLayer[1], nextLayer[0]);",
            "        }",
            "    };",
        ]
    )

    assert expected == actual


@pytest.mark.unit
@pytest.mark.cartographer
def test_js_first_layer_listener():
    """test cartographer.js_first_layer_listener"""

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

    actual = c.js_first_layer_listener(layer_dict)

    expected = '    test.on("load", loadNextLayer);'

    assert expected == actual


@pytest.mark.unit
@pytest.mark.cartographer
def test_colors_js():
    """test cartographer.colors_js"""
    expected = "\n".join(
        [
            "    let colors = [",
            '        "#4C72B0",',
            '        "#DD8452",',
            '        "#55A868",',
            '        "#C44E52",',
            '        "#8172B3",',
            '        "#937860",',
            '        "#DA8BC3",',
            '        "#8C8C8C",',
            '        "#CCB974",',
            '        "#64B5CD",',
            "    ];",
        ]
    )

    assert expected == c.colors_js()


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
            "    L.CRS.FitsMap = L.extend({}, L.CRS.Simple, {",
            f"        transformation: new L.Transformation(1/{int(2**max_zoom)}, 0, -1/{int(2**max_zoom)}, 256)",
            "    });",
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
            '    var map = L.map("map", {',
            "        crs: L.CRS.FitsMap,",
            "        minZoom: " + str(min_zoom) + ",",
            "        preferCanvas: true,",
            "    });",
        ]
    )

    assert expected_map_js == acutal_map_js


@pytest.mark.unit
@pytest.mark.cartographer
def test_leaflet_map_set_view():
    """test cartographer.leaflet_map_set_view"""

    actual = c.leaflet_map_set_view()
    expected = "\n".join(
        [
            '    if (urlParam("zoom")==null){',
            '        map.fitWorld({"maxZoom":map.getMinZoom()});',
            "    } else{",
            "        panFromUrl(map);",
            "    }",
        ]
    )

    assert expected == actual


@pytest.mark.unit
@pytest.mark.cartographer
def test_markers_to_js():
    """test cartographer.markers_to_js"""

    name = "test_0.cat.js"

    actual_marker_js = c.markers_to_js([name])

    expected_marker_js = "\n".join(
        [
            "    // catalogs ================================================================",
            "    // a preset list of colors to use for markers in different catalogs",
            "    let colors = [",
            '        "#4C72B0",',
            '        "#DD8452",',
            '        "#55A868",',
            '        "#C44E52",',
            '        "#8172B3",',
            '        "#937860",',
            '        "#DA8BC3",',
            '        "#8C8C8C",',
            '        "#CCB974",',
            '        "#64B5CD",',
            "    ];",
            "",
            "    // each list will hold the markers. if a catalog is sharded then there",
            "    // will be multiple lists in a each top-level list element.",
            "    var markerList = [",
            "        [[]],",
            "    ];",
            "",
            "    // the variables containing the catalog information, this mirrors the",
            "    // structre in `markerList`. the sources in these variables will be",
            "    // converted into markers and then added to the corresponding array",
            "    const collections = [",
            "        [test_cat_var_0],",
            "    ];",
            "    var labels = [",
            "        '<span style=\"color:red\">test</span>::0/'+ 1 +'-0%',",
            "    ];",
            "",
            "    // `collections_idx` is a collection of indexes that can be popped to",
            "    // asynchronously process the catalog data in `collections`",
            "    collection_idx = []",
            "    for (var i = 0; i < collections.length; i++){",
            "        collection_idx.push([...Array(collections[i].length).keys()])",
            "    }",
            "",
            "    // declare markers up here for scope",
            "    var markers = [];",
            "",
            "    // this is a function that returns a callback function for the chunked",
            "    // loading function of markerClusterGroups",
            "    function update_f(i){",
            '        //console.log("update_f", i);',
            "",
            "        // the markerClusterGroups callback function takes three arguments",
            "        // nMarkers: number of markers processed so far",
            "        // total:    total number of markers in shard",
            "        // elapsed:  time elapsed (not used)",
            "        return (nMarkers, total, elasped) => {",
            "",
            "            var completetion = total==0 ? 0 : nMarkers/total;",
            "",
            "            name_tag = layerControl._layers[i].name;",
            '            split_values = name_tag.split("::");',
            "            html_name = split_values[0];",
            "            progress = split_values[1];",
            "",
            '            current_iter = parseInt(progress.split("/")[0]) + Math.floor(completetion);',
            '            total_iter = parseInt(progress.split("/")[1].split("-")[0]);',
            '            html_name = name_tag.split("::")[0];',
            "",
            "            if (completetion==1 && current_iter==total_iter){",
            '                layerControl._layers[i].name = html_name.replace("red", "black");',
            "            }",
            "            else {",
            '                layerControl._layers[i].name = html_name + "::" + current_iter + "/" + total_iter + "-" + Math.floor(completetion*100) + "%";',
            "            }",
            "            layerControl._update();",
            "",
            "            // if we have finished processing move on to the next shard/catalog",
            "            if (completetion==1){",
            "                add_marker_collections_f(i);",
            "            }",
            "        }",
            "    };",
            "",
            "    const panes = [",
            '        "test",',
            "    ];",
            "    panes.forEach(i => {map.createPane(i).style.zIndex = 0;});",
            "",
            "    for (var i = 0; i < panes.length; i++){",
            "        markers.push(",
            "            L.markerClusterGroup({'chunkedLoading':true, 'chunkInterval':50, 'chunkDelay':50, 'chunkProgress':update_f(i), 'clusterPane':panes[i]}),",
            "        );",
            "    }",
            "",
            "    for (var i = 0; i < markers.length; i++){",
            "        layerControl.addOverlay(markers[i], labels[i]);",
            "    }",
            "",
            "    function add_marker_collections(event){",
            "        add_marker_collections_f(collection_idx.length-1);",
            "    }",
            "",
            "    function add_marker_collections_f(i){",
            "        //console.log('i is currently ', i);",
            "        if (i >= 0){",
            "            if (collection_idx[i].length > 0) {",
            "                j = collection_idx[i].pop();",
            "                markers[i].addLayers(markerList[i][j]);",
            "            } else {",
            "                markers[i].options.chunkProgress = null;",
            "                layerControl._update();",
            "",
            "                // this for some reason causes an error, but doesn't seem to",
            "                // affect the map.",
            "                map.getPane(panes[i]).style.zIndex=650;",
            "                markers[i].remove();",
            "",
            "                add_marker_collections_f(i-1);",
            "            }",
            "        }",
            "    };",
            "",
            "    for (i = 0; i < collections.length; i++){",
            "        collection = collections[i];",
            "        //console.log(i, collection);",
            "",
            "        for (ii = 0; ii < collection.length; ii++){",
            "            collec = collection[ii];",
            "            for (j = 0; j < collec.length; j++){",
            "                src = collec[j];",
            "",
            "                var width = (((src.widest_col * 10) * src.n_cols) + 10).toString() + 'em';",
            "                var include_img = src.include_img ? 2 : 1;",
            "                var height = ((src.n_rows + 1) * 15 * (include_img)).toString() + 'em';",
            "",
            '                let p = L.popup({ maxWidth: "auto" })',
            "                         .setLatLng([src.y, src.x])",
            '                         .setContent("<iframe src=\'catalog_assets/" + src.cat_path + "/" + src.catalog_id + ".html\' width=\'" + width + "\' height=\'" + height + "\'></iframe>");',
            "",
            "                let marker;",
            "                if (src.a==-1){",
            "                    marker = L.circleMarker([src.y, src.x], {",
            "                        catalog_id: panes[i] + ':' + src.catalog_id + ':',",
            "                        color: colors[i % colors.length]",
            "                    }).bindPopup(p);",
            "                } else {",
            "                    marker = L.ellipse([src.y, src.x], [src.a, src.b], (src.theta * (180/Math.PI) * -1), {",
            "                        catalog_id: panes[i] + ':' + src.catalog_id + ':',",
            "                        color: colors[i % colors.length]",
            "                    }).bindPopup(p);",
            "                }",
            "",
            "                markerList[i][ii].push(marker);",
            "            }",
            "        }",
            "    }",
            "",
            '    map.on("load", add_marker_collections);',
            "    var marker_layers = L.layerGroup(markers);",
            "    // =========================================================================",
        ]
    )

    assert expected_marker_js == actual_marker_js


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

    test_file = "test_0.js"

    acutal_js = c.build_conditional_js(helpers.TEST_PATH, [test_file])

    expected_js = "\n".join(
        [
            "    <script src='https://unpkg.com/leaflet.markercluster@1.4.1/dist/leaflet.markercluster-src.js' crossorigin=''></script>",
            "    <script src='https://unpkg.com/leaflet-search@2.9.8/dist/leaflet-search.src.js' crossorigin=''></script>",
            "    <script src='js/test_0.js'></script>",
            "    <script src='js/l.ellipse.min.js'></script>",
        ]
    )

    helpers.tear_down()

    assert expected_js == acutal_js


@pytest.mark.unit
@pytest.mark.cartographer
def test_leaflet_layer_control_declaration():
    """test cartographer.leaflet_layer_control_declaration"""

    actual = c.leaflet_layer_control_declaration()
    expected = "    var layerControl = L.control.layers();"

    assert expected == actual


@pytest.mark.unit
@pytest.mark.cartographer
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
            "    <title>{}</title>".format(title),
            '    <meta charset="utf-8" />',
            '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            '    <link rel="shortcut icon" type="image/x-icon" href="docs/images/favicon.ico" />',
            '    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.3.4/dist/leaflet.css" integrity="sha512-puBpdR0798OZvTTbP4A8Ix/l+A4dHDD0DGqYW6RQ+9jxkRFclaxxQb/SJAWZfWAkuyeQUytO7+7N4QKrDh+drA==" crossorigin=""/>',
            extra_css,
            "    <script src='https://unpkg.com/leaflet@1.3.4/dist/leaflet.js' integrity='sha512-nMMmRyTVoLYqjP9hrbed9S+FzjZHW5gY1TWCHA5ckwXZBadntCNs8kEqAWdrb9O7rxbCaA4lKTIWjDXZxflOcA==' crossorigin=''></script>",
            extra_js,
            "    <style>",
            "        html, body {",
            "        height: 100%;",
            "        margin: 0;",
            "        }",
            "        #map {",
            "            width: 100%;",
            "            height: 100%;",
            "        }",
            "    </style>",
            "</head>",
            "<body>",
            '    <div id="map"></div>',
            "    <script>",
            js,
            "    </script>",
            "</body>",
            f"<!--Made with fitsmap v{helpers.get_version()}-->",
            "</html>\n",
        ]
    )

    assert expected_html == actual_html


@pytest.mark.unit
@pytest.mark.cartographer
def test_build_digit_to_string():
    """test cartographer.build_digit_to_string"""
    digits = range(10)
    strings = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    ]

    for expected, actual in zip(strings, map(c.digit_to_string, digits)):
        assert expected == actual


@pytest.mark.unit
@pytest.mark.cartographer
def test_build_digit_to_string_fails():
    """test cartographer.build_digit_to_string"""
    digit = -1

    with pytest.raises(ValueError) as excinfo:
        c.digit_to_string(digit)

    assert "Only digits 0-9 are supported" in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.cartographer
def test_make_fname_js_safe_digit():
    """Test the cartographer.make_fname_js_safe functions."""

    unsafe = "123"
    expected = "one23"

    assert expected == c.make_fname_js_safe(unsafe)


@pytest.mark.unit
@pytest.mark.cartographer
def test_make_fname_js_safe_dot_dash():
    """Test the cartographer.make_fname_js_safe functions."""

    unsafe = "a.b-c"
    expected = "a_dot_b_c"

    assert expected == c.make_fname_js_safe(unsafe)


@pytest.mark.unit
@pytest.mark.cartographer
def test_make_fname_js_safe_no_change():
    """Test the cartographer.make_fname_js_safe functions."""

    safe = "abc"
    expected = "abc"

    assert expected == c.make_fname_js_safe(safe)


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

    c.chart(out_dir, title, [map_layer_names], [marker_file_names], wcs)

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

    c.chart(out_dir, title, [map_layer_names], [marker_file_names], wcs)

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
