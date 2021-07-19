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

"""Helper functions for creating a leaflet JS HTML map."""

# ******************************************************************************
# Designed for internal use. Any method/variable can be deprecated/changed
# without consideration.
# ******************************************************************************

import os
import shutil
import string
from itertools import repeat
from functools import partial, reduce
from typing import Dict, List

import numpy as np
from astropy.wcs import WCS

import fitsmap

# None defaults here mean width/height will be
# calculated based on table column properties
MARKER_HTML_WIDTH = None
MARKER_HTML_HEIGHT = None

MARKER_SEARCH_JS = "\n".join(
    [
        "    // search function =========================================================",
        "    function searchHelp(e) {",
        "        map.setView(e.latlng, 4);",
        "        e.layer.addTo(map);",
        "    };",
        "",
        "    var searchBar = L.control.search({",
        "        layer: marker_layers,",
        "        initial: false,",
        "        propertyName: 'catalog_id',",
        "        textPlaceholder: 'Enter catalog_id ID',",
        "        hideMarkerOnCollapse: true,",
        "    });",
        "",
        "    searchBar.on('search:locationfound', searchHelp);",
        "    searchBar.addTo(map);",
        "    // =========================================================================",
    ]
)

LAYER_ATTRIBUTION = "<a href='https://github.com/ryanhausen/fitsmap'>FitsMap</a>"


def chart(
    out_dir: str,
    title: str,
    map_layer_names: List[str],
    marker_file_names: List[str],
    wcs: WCS,
) -> None:
    """Creates an HTML file containing a leaflet js map using the given params.

    ****************************************************************************
    * Designed for internal use. Any method/variable can be deprecated/changed *
    * without consideration.                                                   *
    ****************************************************************************
    """

    # convert layer names into a single javascript string
    layer_zooms = lambda l: list(map(int, os.listdir(os.path.join(out_dir, l))))
    zooms = reduce(lambda x, y: x + y, list(map(layer_zooms, map_layer_names)))
    zooms = [0] if len(zooms) == 0 else zooms
    convert_layer_name_func = partial(layer_name_to_dict, min(zooms), max(zooms))
    layer_dicts = list(map(convert_layer_name_func, map_layer_names))

    # build leafletjs js string
    js_crs = leaflet_crs_js(layer_dicts)

    js_layer_control_declaration = leaflet_layer_control_declaration()

    js_first_layer = layer_dict_to_str(layer_dicts[0])

    js_async_layers_str = js_async_layers(layer_dicts[1:])

    js_markers = markers_to_js(marker_file_names) if marker_file_names else ""

    js_map = leaflet_map_js(layer_dicts)

    js_marker_search = MARKER_SEARCH_JS if marker_file_names else ""

    js_add_layer_control = add_layer_control(layer_dicts[0])

    js_leaflet_wcs = leaflet_wcs_js(wcs)

    js_set_map = leaflet_map_set_view()

    js = "\n".join(
        [
            js_crs,
            js_layer_control_declaration,
            js_map,
            js_first_layer,
            js_async_layers_str,
            js_load_next_layer(),
            js_first_layer_listener(layer_dicts[0]),
            js_markers,
            js_marker_search,
            js_add_layer_control,
            js_leaflet_wcs,
            js_set_map,
        ]
    )

    extra_js = (
        build_conditional_js(out_dir, marker_file_names) if marker_file_names else ""
    )

    extra_css = build_conditional_css(out_dir) if marker_file_names else ""

    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(build_html(title, js, extra_js, extra_css))


def layer_name_to_dict(min_zoom: int, max_zoom: int, name: str) -> dict:
    """Convert layer name to dict for conversion."""

    return dict(
        directory=name + "/{z}/{y}/{x}.png",
        name=name,
        min_zoom=min_zoom,
        max_zoom=max_zoom + 5,
        max_native_zoom=max_zoom,
    )


def layer_dict_to_str(layer: dict) -> str:
    """Convert layer dict to layer str for including in HTML file."""

    layer_str = [
        "    const " + layer["name"],
        ' = L.tileLayer("' + layer["directory"] + '"',
        ", { ",
        'attribution:"' + LAYER_ATTRIBUTION + '",',
        "minZoom: " + str(layer["min_zoom"]) + ",",
        "maxZoom: " + str(layer["max_zoom"]) + ",",
        "maxNativeZoom: " + str(layer["max_native_zoom"]) + ",",
        "}).addTo(map);",
    ]

    return "".join(layer_str)


def add_layer_control(layer: dict):
    js = [
        f'    layerControl.addBaseLayer({layer["name"]}, "{layer["name"]}");',
        "    layerControl.addTo(map);",
    ]

    return "\n".join(js)


# TODO: This should be factored into a map call to a single format function
def js_async_layers(async_layers: List[Dict]) -> str:
    """Converts all layers after the first one into layers that area loaded async"""

    def layer_fmt(l: dict):
        return "".join(
            [
                '        ["',
                l["name"],
                '", L.tileLayer("',
                l["directory"],
                '",{ attribution:"',
                LAYER_ATTRIBUTION,
                '", minZoom:',
                str(l["min_zoom"]),
                ", maxZoom:",
                str(l["max_zoom"]),
                ", maxNativeZoom:",
                str(l["max_native_zoom"]),
                "})],",
            ]
        )

    js = ["    asyncLayers = [", *list(map(layer_fmt, async_layers)), "    ];"]

    return "\n".join(js)


def js_load_next_layer() -> str:
    return "\n".join(
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


def js_first_layer_listener(first_layer: dict) -> str:
    return f'    {first_layer["name"]}.on("load", loadNextLayer);'


def colors_js() -> str:
    js = [
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

    return "\n".join(js)


def leaflet_crs_js(tile_layers: List[dict]) -> str:
    max_zoom = max(map(lambda t: t["max_native_zoom"], tile_layers))

    scale_factor = int(2 ** max_zoom)

    js = [
        "    L.CRS.FitsMap = L.extend({}, L.CRS.Simple, {",
        f"        transformation: new L.Transformation(1/{scale_factor}, 0, -1/{scale_factor}, 256)",
        "    });",
    ]

    return "\n".join(js)


def leaflet_map_js(tile_layers: List[dict]):
    js = [
        '    var map = L.map("map", {',
        "        crs: L.CRS.FitsMap,",
        "        minZoom: " + str(max(map(lambda t: t["min_zoom"], tile_layers))) + ",",
        "        preferCanvas: true,",
        "    });",
    ]

    return "\n".join(js)


def leaflet_map_set_view():
    js = [
        '    if (urlParam("zoom")==null){',
        '        map.fitWorld({"maxZoom":map.getMinZoom()});',
        "    } else{",
        "        panFromUrl(map);",
        "    }",
    ]

    return "\n".join(js)


# TODO: Maybe break this up into handling single marker files?
def markers_to_js(marker_file_names: List[str]) -> str:
    """Convert marker file names into marker javascript for the HTML file."""

    deshard_name = lambda s: "_".join(s.replace(".cat.js", "").split("_")[:-1])
    desharded_names = list(map(deshard_name, marker_file_names))
    unique_names = set(sorted(desharded_names))
    shard_counts = list(map(desharded_names.count, unique_names))
    names_counts = list(zip(unique_names, shard_counts))

    marker_list_f = lambda n: "        [" + ",".join(repeat("[]", n)) + "],"

    def expand_name_counts(name_count):
        name, cnt = name_count
        fmt = lambda i: f"{make_fname_js_safe(name)}_cat_var_{i}"

        return "        [" + ", ".join(map(fmt, range(cnt))) + "],"

    def convert_name_count_to_label(name_count):
        name, cnt = name_count
        return "        '<span style=\"color:red\">{}</span>::0/'+ {} +'-0%',".format(
            name, cnt
        )

    if MARKER_HTML_WIDTH:
        var_marker_width = f"                var width = '{MARKER_HTML_WIDTH}';"
    else:
        var_marker_width = "                var width = (((src.widest_col * 10) * src.n_cols) + 10).toString() + 'em';"

    if MARKER_HTML_HEIGHT:
        var_marker_height = f"                var height = '{MARKER_HTML_HEIGHT}';"
    else:
        var_marker_height = "                var height = ((src.n_rows + 1) * 15 * (include_img)).toString() + 'em';"

    js = [
        "    // catalogs ================================================================",
        "    // a preset list of colors to use for markers in different catalogs",
        colors_js(),
        "",
        "    // each list will hold the markers. if a catalog is sharded then there",
        "    // will be multiple lists in a each top-level list element.",
        "    var markerList = [",
        *list(map(marker_list_f, shard_counts)),
        "    ];",
        "",
        "    // the variables containing the catalog information, this mirrors the",
        "    // structre in `markerList`. the sources in these variables will be",
        "    // converted into markers and then added to the corresponding array",
        "    const collections = [",
        *list(map(expand_name_counts, names_counts)),
        "    ];" "",
        "    var labels = [",
        *list(map(convert_name_count_to_label, names_counts)),
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
        *list(map(lambda s: f'        "{s}",', unique_names)),
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
        var_marker_width,
        "                var include_img = src.include_img ? 2 : 1;",
        var_marker_height,
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

    return "\n".join(js)


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


def leaflet_wcs_js(img_wcs: WCS) -> str:
    """Functions for translating image and sky coordinates."""

    wcs_js = "\n".join(
        [
            "",
            "    // WCS functionality =======================================================",
            "    const is_ra_dec = _IS_RA_DEC;",
            "    const crpix = _CRPIX;",
            "    const crval = _CRVAL;",
            "    const cdmatrix = _CD;",
            "",
            "    urlParam = function(name){",
            "        // Parse parameters from window.location,",
            "        // e.g., .../index.html?zoom=8",
            "        // urlParam(zoom) = 8",
            "        var results = new RegExp('[\?&]' + name + '=([^&#]*)').exec(window.location.href);",
            "        if (results==null){",
            "            return null;",
            "        }",
            "        else{",
            "            return decodeURI(results[1]) || 0;",
            "        }",
            "    }",
            "",
            "    pixToSky = function(xy){",
            "        // Convert from zero-index pixel to sky coordinate assuming",
            "        // simple North-up WCS",
            "        if (xy.hasOwnProperty('lng')){",
            "            var dx = xy.lng - crpix[0] + 1;",
            "            var dy = xy.lat - crpix[1] + 1;",
            "        } else {",
            "            var dx = xy[0] - crpix[0] + 1;",
            "            var dy = xy[1] - crpix[1] + 1;",
            "        }",
            "        var dra = dx * cdmatrix[0][0];",
            "        var ddec = dy * cdmatrix[1][1];",
            "        // some catalogs are stored in image coords x/y, not ra/dec. When",
            "        // `is_ra_dec`==1 we are doing calculation in ra/dec when `is_ra_dec`==0",
            "        // then we're working in image coords and so multiply by 0 so",
            "        // cos(0)==1",
            "        var ra = crval[0] + dra / Math.cos(crval[1]/180*3.14159 * is_ra_dec);",
            "        var dec = crval[1] + ddec;",
            "        return [ra, dec];",
            "    }",
            "",
            "    skyToPix = function(rd){",
            "        // Convert from sky to zero-index pixel coordinate assuming",
            "        // simple North-up WCS",
            "        var dx = (rd[0] - crval[0]) * Math.cos(crval[1]/180*3.14159 * is_ra_dec);",
            "        var dy = (rd[1] - crval[1]);",
            "        var x = crpix[0] - 1 + dx / cdmatrix[0][0];",
            "        var y = crpix[1] - 1 + dy / cdmatrix[1][1];",
            "        return [x,y];",
            "    }",
            "",
            "    skyToLatLng = function(rd){",
            "        // Convert from sky to Leaflet.latLng coordinate assuming",
            "        // simple North-up WCS",
            "        var xy = skyToPix(rd);",
            "        return L.latLng(xy[1], xy[0]);",
            "    }",
            "",
            "    panToSky = function(rd, zoom, map){",
            "        // Pan map to celestial coordinates",
            "        var ll = skyToLatLng(rd)",
            "        map.setZoom(zoom);",
            "        map.panTo(ll, zoom);",
            "        //console.log('pan to: ' + rd + ' / ll: ' + ll.lng + ',' + ll.lat);",
            "    }",
            "",
            "    panFromUrl = function(map){",
            "        // Pan map based on ra/dec/[zoom] variables in location bar",
            "        var ra = urlParam('ra');",
            "        var dec = urlParam('dec');",
            "        var zoom = urlParam('zoom') || map.getMinZoom();",
            "        if ((ra !== null) & (dec !== null)) {",
            "            panToSky([ra,dec], zoom, map);",
            "        } else {",
            "            // Pan to crval",
            "            panToSky(crval, zoom, map);",
            "        }",
            "    }",
            "",
            "    updateLocationBar = function(){",
            "        var rd = pixToSky(map.getCenter());",
            "        //console.log(rd);",
            "        var params = 'ra=' + rd[0].toFixed(7);",
            "        params += '&dec=' + rd[1].toFixed(7);",
            "        params += '&zoom=' + map.getZoom();",
            "        //console.log(params);",
            "        var param_url = window.location.href.split('?')[0] + '?' + params;",
            "        window.history.pushState('', '', param_url);",
            "    }",
            "",
            "    map.on('moveend', updateLocationBar);",
            "    map.on('zoomend', updateLocationBar);",
            "    // WCS functionality =======================================================",
        ]
    )

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


def build_conditional_js(out_dir: str, marker_file_names: List[str]) -> str:

    support_dir = os.path.join(os.path.dirname(__file__), "support")
    out_js_dir = os.path.join(out_dir, "js")

    local_js_files = list(
        filter(lambda f: os.path.splitext(f)[1] == ".js", os.listdir(support_dir))
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

    leaflet_js = [
        "    <script src='https://unpkg.com/leaflet.markercluster@1.4.1/dist/leaflet.markercluster-src.js' crossorigin=''></script>",
        "    <script src='https://unpkg.com/leaflet-search@2.9.8/dist/leaflet-search.src.js' crossorigin=''></script>",
    ]

    js_string = "    <script src='js/{}'></script>"
    local_js = list(
        map(lambda s: js_string.format(s), marker_file_names + local_js_files)
    )

    return "\n".join(leaflet_js + local_js)


def leaflet_layer_control_declaration():
    return "    var layerControl = L.control.layers();"


def build_html(title: str, js: str, extra_js: str, extra_css: str) -> str:
    html = [
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
        f"<!--Made with fitsmap v{get_version()}-->",
        "</html>\n",
    ]

    return "\n".join(html)


def get_version():
    with open(os.path.join(fitsmap.__path__[0], "__version__.py"), "r") as f:
        return f.readline().strip().replace('"', "")


def digit_to_string(digit: int) -> str:
    if digit == 0:
        return "zero"
    elif digit == 1:
        return "one"
    elif digit == 2:
        return "two"
    elif digit == 3:
        return "three"
    elif digit == 4:
        return "four"
    elif digit == 5:
        return "five"
    elif digit == 6:
        return "six"
    elif digit == 7:
        return "seven"
    elif digit == 8:
        return "eight"
    elif digit == 9:
        return "nine"
    else:
        raise ValueError("Only digits 0-9 are supported")


def make_fname_js_safe(fname: str) -> str:
    """Converts a string filename to a javascript safe identifier."""

    if fname[0] in string.digits:
        adj_for_digit = digit_to_string(int(fname[0])) + fname[1:]
    else:
        adj_for_digit = fname

    return adj_for_digit.replace(".", "_dot_").replace("-", "_")
