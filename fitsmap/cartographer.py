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
import string
from itertools import count, repeat
from functools import partial, reduce
from typing import Dict, List, Tuple

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
    img_xy: Tuple[int,int],
) -> None:
    """Creates an HTML file containing a leaflet js map using the given params.

    ****************************************************************************
    * Designed for internal use. Any method/variable can be deprecated/changed *
    * without consideration.                                                   *
    ****************************************************************************
    """
    # TODO need to get these numbers
    img_x, img_y = img_xy

    # convert layer names into a single javascript string
    layer_zooms = lambda l: list(map(int, os.listdir(os.path.join(out_dir, l))))
    zooms = reduce(lambda x, y: x + y, list(map(layer_zooms, map_layer_names)))
    zooms = [0] if len(zooms) == 0 else zooms
    convert_layer_name_func = partial(layer_name_to_dict, min(zooms), max(zooms))
    layer_dicts = list(map(convert_layer_name_func, map_layer_names))

    # generated javascript =====================================================
    with open(os.path.join(out_dir, "js", "worker.js"), "w") as f:
        f.write(build_worker_js(img_x, img_y))

    with open(os.path.join(out_dir, "js", "urlCoords.js"), "w") as f:
        f.write(build_urlCoords_js(wcs))

    with open(os.path.join(out_dir, "js", "index.js"), "w") as f:
        f.write(build_index_js(layer_dicts, marker_file_names))
    # generated javascript =====================================================

    # HTML file contents =======================================================
    extra_js = (
        build_conditional_js(out_dir) if marker_file_names else ""
    )

    extra_css = build_conditional_css(out_dir) if marker_file_names else ""

    move_supporting_imgs(out_dir)

    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(build_html(title, extra_js, extra_css))
    # HTML file contents =======================================================


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


def colors_js() -> str:
    js = [
        "const colors = [",
        '    "#4C72B0",',
        '    "#DD8452",',
        '    "#55A868",',
        '    "#C44E52",',
        '    "#8172B3",',
        '    "#937860",',
        '    "#DA8BC3",',
        '    "#8C8C8C",',
        '    "#CCB974",',
        '    "#64B5CD",',
        "];",
    ]

    return "\n".join(js)


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
            'var map = L.map("map", {',
            "    crs: L.CRS.FitsMap,",
            "    minZoom: " + str(max(map(lambda t: t["min_zoom"], tile_layers))) + ",",
            "    preferCanvas: true,",
            f"    layers: [{tile_layers[0]['name']}]",
            "});",
        ]
    )

    return js


def marker_filenames_to_js(marker_file_names: List[str], num_image_layers:int) -> str:

    deshard_name = lambda s: "_".join(s.replace(".cat.json", "").split("_")[:-1])
    desharded_names = list(map(deshard_name, marker_file_names))
    unique_names = set(sorted(desharded_names))
    shard_counts = list(map(desharded_names.count, unique_names))
    names_counts = list(zip(unique_names, shard_counts))
    n_catalogs = len(unique_names)

    def make_fname_lists(name_count):
        return (
            "    ["
            + ",".join(
                map(lambda i: f'"../json/{name_count[0]}_{i}.cat.json"', range(name_count[1]))
            )
            + "]"
        )

    marker_display_mask = "<span class='loading'><img class='loading' src='img/load.gif'></img>{}::Fetching[1/{}]</span>"
    def name_to_layer(idx_namecount):
        return '    [L.geoJSON(null, {{pointToLayer: (feat, latlng) => createClusterIcon({}, feat, latlng),}}), "{}"],'.format(
            idx_namecount[0],
            marker_display_mask.format(*idx_namecount[1])
        )


    js = "\n".join(
        [
            "const catalogFileNames = [",
            *list(map(make_fname_lists, names_counts)),
            "];",
            "",
            "const catalogMarkers = [",
            *list(map(name_to_layer, zip(count(), names_counts))),
            "];",
            "",
            f"const catalogsLoaded = [{','.join(repeat('false', n_catalogs))}];" "",
            "const catalogWorkers = [",
            *list(repeat('    new Worker("js/worker.js"),', n_catalogs)),
            "];",
            "",
            f"const nImageLayers = {num_image_layers};",
            "for (let i = 0; i < catalogWorkers.length; i++){",
            "    layerControl.addOverlay(...catalogMarkers[i]);",
            "    catalogWorkers[i].onmessage = function(e) {",
            "        if (e.data.ready) {",
            "            catalogsLoaded[i] = true;",
            "",
            "            let catalogText = layerControl._layers[i + nImageLayers].name;",
            '            catalogText = catalogText.replaceAll(/loading/g, "loading-complete")',
            '                                     .replace("::Parsing", "");',
            "            layerControl._layers[i + nImageLayers].name = catalogText;",
            "            layerControl._update();"
            "",
            "            update();",
            "        } else if (e.data.progress){",
            "            let catalogText = layerControl._layers[i + nImageLayers].name;",
           r"            let catalogIndex = parseInt(catalogText.match(/(?<=\[)(\d)(?=\/\d\])/)[0]);",
           r"            let catlogTotal = parseInt(catalogText.match(/(?<=\[\d\/)(\d)(?=\])/)[0]);",
           r"            catalogText = catalogText.replace(/(?<=\[)(\d)(?=\/\d\])/, (catalogIndex+1).toString()); // added",
            "            if (catalogIndex+1>catlogTotal){",
            '                catalogText = catalogText.replace("::Fetching", "::Parsing")',
           r'                                         .replace(/\[\d\/\d]/, "");',
            "            }",
            "            layerControl._layers[i + nImageLayers].name = catalogText;",
            "            layerControl._update();",
            "        } else if (e.data.expansionZoom){",
            "            map.flyTo(e.data.center, e.data.expansionZoom);",
            "        } else {",
            "            catalogMarkers[i][0].clearLayers();",
            "            catalogMarkers[i][0].addData(e.data);",
            "        }",
            "    };",
            "",
            '    catalogWorkers.onerror = function(e) { console.log("ERROR:", e); }',
            "    catalogWorkers[i].postMessage({fileNames: catalogFileNames[i]});",
            "}",
            "" "function createClusterIcon(colorIdx, feature, latlng) {",
            "",
            "    // create an icon for a single source",
            "    if (!feature.properties.cluster){",
            "        const src = feature.properties;",
            '        const width = (((src.widest_col * 10) * src.n_cols) + 10).toString() + "em";',
            "        const include_img = src.include_img ? 2 : 1;",
            '        const height = ((src.n_rows + 1) * 15 * (include_img)).toString() + "em";',
            "",
            '        const p = L.popup({ maxWidth: "auto" })',
            "                 .setLatLng(latlng)",
            '                 .setContent("<iframe src=\'catalog_assets/" + src.cat_path + "/" + src.catalog_id + ".html\' width=\'" + width + "\' height=\'" + height + "\'></iframe>");',
            "",
            "",
            "        if (src.a==-1){",
            "            return L.circleMarker(latlng, {",
            "                color: colors[colorIdx % colors.length]",
            "            }).bindPopup(p);",
            "        } else {",
            "            return L.ellipse(latlng, [src.a, src.b], (src.theta * (180/Math.PI) * -1), {",
            "                color: colors[colorIdx % colors.length]",
            "            }).bindPopup(p);",
            "        }",
            "",
            "    }",
            "",
            "    // Create an icon for a cluster",
            "    const count = feature.properties.point_count;",
            "    const size =",
            '        count < 100 ? "small" :',
            '        count < 1000 ? "medium" : "large";',
            "    const icon = L.divIcon({",
            "        html: `<div><span>${  feature.properties.point_count_abbreviated  }</span></div>`,",
            "        className: `marker-cluster marker-cluster-${size}`,",
            "        iconSize: L.point(40, 40)",
            "    });",
            "",
            "    return L.marker(latlng, {icon});",
            "}",
            "",
            "function update() {",
            "    all_loaded  = (accumulator, currentValue) => accumulator && currentValue;",
            "    if (!all_loaded(catalogsLoaded, true)) return;",
            "",
            "    const bounds = map.getBounds();",
            "    catalogWorkers.forEach(worker => {",
            "        worker.postMessage({",
            "            bbox: [bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()],",
            "            zoom: map.getZoom()",
            "        });",
            "    });",
            "}",
        ]
    )

    return js


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
        map(lambda s: js_string.format(s), local_js_files)
    )

    return "\n".join(leaflet_js + local_js)


def move_supporting_imgs(out_dir: str) -> None:
    support_dir = os.path.join(os.path.dirname(__file__), "support")
    out_img_dir = os.path.join(out_dir, "img")

    image_exts = [".png", ".jpg", ".gif", ".jpeg"]

    local_image_files = list(
        filter(lambda f: os.path.splitext(f)[1] in image_exts, os.listdir(support_dir))
    )

    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)

    all(
        map(
            lambda f: shutil.copy2(
                os.path.join(support_dir, f), os.path.join(out_img_dir, f)
            ),
            local_image_files,
        )
    )

def leaflet_layer_control_declaration(layer_dicts: List[Dict]) -> str:
    layer_label_pairs = ",".join(
        list(map(lambda l: '"{0}": {0}'.format(l["name"]), layer_dicts))
    )

    return f"const layerControl = L.control.layers({{ {layer_label_pairs} }}).addTo(map);"


def build_worker_js(img_x: int, img_y: int):
    js = "\n".join(
        [
            "// A catalog worker. The catalog worker is responsible for managing a single",
            "// catalog in fitsmap.",
            "// Based on https://github.com/mapbox/supercluster/blob/master/demo/worker.js",
            "",
            'importScripts("supercluster.js");',
            "",
            "const now = Date.now();",
            "const catalogSources = {",
            '    "type": "FeatureCollection",',
            '    "features": []',
            "};",
            "",
            "index = new Supercluster({",
            "    log: true,",
            "    radius: 128,",
            "    extent: 256,",
            "    maxZoom: 10,",
            "    alternateCRS: true",
            "});",
            f"index.lngX = function lngX(lng) {{ return lng / {img_x}; }}",
            f"index.latY = function latY(lat) {{ return lat / {img_y}; }}",
            f"index.xLng = function xLng(x) {{ return x * {img_x}; }}",
            f"index.yLat = function yLat(y) {{ return y * {img_y}; }}",
            "",
            "let ready = false;",
            "self.onmessage = function (e) {",
            "    // first run, we're loading data",
            "    if (e.data.fileNames) {",
            "        loadFiles(e.data.fileNames, () => {",
            "            console.log(`loaded ${  catalogSources.features.length  } points JSON in ${  (Date.now() - now) / 1000  }s`);",
            "",
            "            index.load(catalogSources.features);",
            "",
            "            ready = true;",
            '            //console.log("TILE:000", index.getTile(0, 0, 0));',
            "",
            "            postMessage({ready: true});",
            "        });",
            "    // get a cluster expansion clicked zoom",
            "    } else if (ready && e.data.getClusterExpansionZoom) {",
            "        postMessage({",
            "            expansionZoom: index.getClusterExpansionZoom(e.data.getClusterExpansionZoom),",
            "            center: e.data.center",
            "        });",
            "    // general zooming/panning",
            "    } else if (ready && e.data) {",
            "        postMessage(index.getClusters(e.data.bbox, e.data.zoom));",
            "    }",
            "};",
            "",
            "function loadFiles(fileNames, callback){",
            "    // all done, build index",
            "    if (fileNames.length == 0) {",
            "        callback();",
            "    // load the next file",
            "    } else {",
            "        fname = fileNames.pop();",
            "        loadData(fname, (data) => {",
            "            data.features.forEach((d) => catalogSources.features.push(d));",
            "            postMessage({progress:1});",
            "            loadFiles(fileNames, callback);",
            "        });",
            "    }",
            "}",
            "",
            "// maybe switch this to fetch as in",
            "// https://github.com/mapbox/supercluster/pull/170/files/95f293f60e14b1dc5f368eb6a5d8ba8e424bf387..5d727476a67ae94375838953eea65530df708d67",
            "function loadData(url, callback){",
            "    const xhr = new XMLHttpRequest();",
            '    xhr.open("GET", url, true);',
            '    xhr.responseType = "json";',
            '    xhr.setRequestHeader("Accept", "application/json");',
            "    xhr.onload = function () {",
            "        if (xhr.readyState === 4 && xhr.status >= 200 && xhr.status < 300 && xhr.response) {",
            "            callback(xhr.response);",
            "        }",
            "    };",
            "    xhr.send();",
            "}",
        ]
    )

    return js


def build_urlCoords_js(img_wcs: WCS) -> str:
    wcs_js = "\n".join(
        [
            "const is_ra_dec = _IS_RA_DEC;",
            "const crpix = _CRPIX;",
            "const crval = _CRVAL;",
            "const cdmatrix = _CD;",
            "",
            "urlParam = function(name){",
            "    // Parse parameters from window.location,",
            "    // e.g., .../index.html?zoom=8",
            "    // urlParam(zoom) = 8",
            "    var results = new RegExp('[\?&]' + name + '=([^&#]*)').exec(window.location.href);",
            "    if (results==null){",
            "        return null;",
            "    }",
            "    else{",
            "        return decodeURI(results[1]) || 0;",
            "    }",
            "}",
            "",
            "pixToSky = function(xy){",
            "    // Convert from zero-index pixel to sky coordinate assuming",
            "    // simple North-up WCS",
            "    if (xy.hasOwnProperty('lng')){",
            "        var dx = xy.lng - crpix[0] + 1;",
            "        var dy = xy.lat - crpix[1] + 1;",
            "    } else {",
            "        var dx = xy[0] - crpix[0] + 1;",
            "        var dy = xy[1] - crpix[1] + 1;",
            "    }",
            "    var dra = dx * cdmatrix[0][0];",
            "    var ddec = dy * cdmatrix[1][1];",
            "    // some catalogs are stored in image coords x/y, not ra/dec. When",
            "    // `is_ra_dec`==1 we are doing calculation in ra/dec when `is_ra_dec`==0",
            "    // then we're working in image coords and so multiply by 0 so",
            "    // cos(0)==1",
            "    var ra = crval[0] + dra / Math.cos(crval[1]/180*3.14159 * is_ra_dec);",
            "    var dec = crval[1] + ddec;",
            "    return [ra, dec];",
            "}",
            "",
            "skyToPix = function(rd){",
            "    // Convert from sky to zero-index pixel coordinate assuming",
            "    // simple North-up WCS",
            "    var dx = (rd[0] - crval[0]) * Math.cos(crval[1]/180*3.14159 * is_ra_dec);",
            "    var dy = (rd[1] - crval[1]);",
            "    var x = crpix[0] - 1 + dx / cdmatrix[0][0];",
            "    var y = crpix[1] - 1 + dy / cdmatrix[1][1];",
            "    return [x,y];",
            "}",
            "",
            "skyToLatLng = function(rd){",
            "    // Convert from sky to Leaflet.latLng coordinate assuming",
            "    // simple North-up WCS",
            "    var xy = skyToPix(rd);",
            "    return L.latLng(xy[1], xy[0]);",
            "}",
            "",
            "panToSky = function(rd, zoom, map){",
            "    // Pan map to celestial coordinates",
            "    var ll = skyToLatLng(rd)",
            "    map.setZoom(zoom);",
            "    map.panTo(ll, zoom);",
            "    //console.log('pan to: ' + rd + ' / ll: ' + ll.lng + ',' + ll.lat);",
            "}",
            "",
            "panFromUrl = function(map){",
            "    // Pan map based on ra/dec/[zoom] variables in location bar",
            "    var ra = urlParam('ra');",
            "    var dec = urlParam('dec');",
            "    var zoom = urlParam('zoom') || map.getMinZoom();",
            "    if ((ra !== null) & (dec !== null)) {",
            "        panToSky([ra,dec], zoom, map);",
            "    } else {",
            "        // Pan to crval",
            "        panToSky(crval, zoom, map);",
            "    }",
            "}",
            "",
            "updateLocationBar = function(){",
            "    var rd = pixToSky(map.getCenter());",
            "    //console.log(rd);",
            "    var params = 'ra=' + rd[0].toFixed(7);",
            "    params += '&dec=' + rd[1].toFixed(7);",
            "    params += '&zoom=' + map.getZoom();",
            "    //console.log(params);",
            "    var param_url = window.location.href.split('?')[0] + '?' + params;",
            "    window.history.pushState('', '', param_url);",
            "}",
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


def build_index_js(layer_dicts: List[Dict], marker_file_names: List[str]) -> str:

    js = "\n".join(
        [
            "// Image layers ================================================================",
            *list(map(layer_dict_to_str, layer_dicts)),
            "// Basic map setup =============================================================",
            leaflet_crs_js(layer_dicts),
            "",
            leaflet_map_js(layer_dicts),
            "",
            leaflet_layer_control_declaration(layer_dicts),
            "",
            "// catalogs layers =============================================================",
            colors_js(),
            "",
            marker_filenames_to_js(marker_file_names, len(layer_dicts)),
            "" "",
            'map.on("moveend", update);',
            'map.on("moveend", updateLocationBar);',
            'map.on("zoomend", updateLocationBar);',
            "",
            'if (urlParam("zoom")==null) {',
            '    map.fitWorld({"maxZoom":map.getMinZoom()});',
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
        '    <script src="js/urlCoords.js"></script>',
        '    <script src="js/index.js"></script>',
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