# MIT License
# Copyright 2020 Ryan Hausen

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
from itertools import repeat
from functools import partial, reduce
from typing import List

MARKER_SEARCH_JS = "\n".join(
    [
        "   var marker_layers = L.layerGroup(markers);",
        "",
        "   function searchHelp(e) {",
        "      map.setView(e.latlng, 4);",
        "      e.layer.addTo(map);",
        "   };",
        "",
        "   var searchBar = L.control.search({",
        "      layer: marker_layers,",
        "      initial: false,",
        "      propertyName: 'catalog_id',",
        "      textPlaceholder: 'Enter catalog_id ID',",
        "      hideMarkerOnCollapse: true,",
        "   });",
        "",
        "   searchBar.on('search:locationfound', searchHelp);",
        "",
        "   searchBar.addTo(map);",
        "",
        "   // hack for turning off markers at start. Throws exception but doesn't",
        "   // crash page. This should be updated when I understand this library better",
        "   for (l of markers){",
        "      l.remove()",
        "   }",
    ]
)

LAYER_ATTRIBUTION = "<a href='https://github.com/ryanhausen/fitsmap'>FitsMap</a>"


def chart(
    out_dir: str, title: str, map_layer_names: List[str], marker_file_names: List[str],
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
    convert_layer_name_func = partial(layer_name_to_dict, min(zooms), max(zooms))
    layer_dicts = list(map(convert_layer_name_func, map_layer_names))

    # build leafletjs js string
    js_layers = "\n".join(map(layer_dict_to_str, layer_dicts))

    js_markers = markers_to_js(marker_file_names) if marker_file_names else ""

    js_crs = leaflet_crs_js(layer_dicts)

    js_map = leaflet_map_js(layer_dicts)

    js_marker_search = MARKER_SEARCH_JS if marker_file_names else ""

    js_base_layers = layers_dict_to_base_layer_js(layer_dicts)

    js_layer_control = layer_names_to_layer_control(marker_file_names)

    js = "\n".join(
        [
            js_crs,
            js_map,
            js_layers,
            js_markers,
            js_marker_search,
            js_base_layers,
            js_layer_control,
        ]
    )

    extra_js = build_conditional_js(marker_file_names) if marker_file_names else ""

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
        "   var " + layer["name"],
        ' = L.tileLayer("' + layer["directory"] + '"',
        ", { ",
        'attribution:"' + LAYER_ATTRIBUTION + '",',
        "minZoom: " + str(layer["min_zoom"]) + ",",
        "maxZoom: " + str(layer["max_zoom"]) + ",",
        "maxNativeZoom: " + str(layer["max_native_zoom"]) + ",",
        "}).addTo(map);",
    ]

    return "".join(layer_str)


def layers_dict_to_base_layer_js(tile_layers: List[dict]):
    js = [
        "   var baseLayers = {",
        *list(map(lambda t: '      "{0}": {0},'.format(t["name"]), tile_layers)),
        "   };",
    ]
    return "\n".join(js)


def layer_names_to_layer_control(marker_file_names: List[str]) -> str:
    if marker_file_names:
        js = [
            "   var overlays = {}",
            "",
            "   for(i = 0; i < markers.length; i++) {",
            "      overlays[labels[i]] = markers[i];",
            "   }",
            "",
            "   var layerControl = L.control.layers(baseLayers, overlays);",
            "   layerControl.addTo(map);",
        ]

        return "\n".join(js)
    else:
        return "   L.control.layers(baseLayers, {}).addTo(map);"


def colors_js() -> str:
    js = [
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

    return "\n".join(js)


def leaflet_crs_js(tile_layers: List[dict]) -> str:
    max_zoom = max(map(lambda t: t["max_native_zoom"], tile_layers))

    scale_factor = int(2 ** max_zoom)

    js = [
        "   L.CRS.FitsMap = L.extend({}, L.CRS.Simple, {",
        f"      transformation: new L.Transformation(1/{scale_factor}, 0, -1/{scale_factor}, 256)",
        "   });",
    ]

    return "\n".join(js)


def leaflet_map_js(tile_layers: List[dict]):
    js = [
        '   var map = L.map("map", {',
        "      crs: L.CRS.FitsMap,",
        "      zoom: " + str(max(map(lambda t: t["min_zoom"], tile_layers))) + ",",
        "      minZoom: " + str(max(map(lambda t: t["min_zoom"], tile_layers))) + ",",
        "      center:[-126, 126],",
        "   });",
    ]

    return "\n".join(js)


# TODO: Maybe break this up into handling single marker files?
def markers_to_js(marker_file_names: List[str]) -> str:
    """Convert marker file names into marker javascript for the HTML file."""

    cluster_text = "      L.markerClusterGroup({ }),"
    marker_list_text = "      [],"

    js = [
        "   var markers = [",
        *list(repeat(cluster_text, len(marker_file_names))),
        "   ];",
        "",
        "   var markerList = [",
        *list(repeat(marker_list_text, len(marker_file_names))),
        "   ];",
        "",
        "   var collections = [",
        *list(
            map(
                lambda s: "      " + s.replace(".cat.js", "_cat_var") + ",",
                marker_file_names,
            )
        ),
        "   ];",
        "",
        "   var labels = [",
        *list(
            map(
                lambda s: "      '" + s.replace(".cat.js", "") + "',",
                marker_file_names,
            )
        ),
        "   ];",
        "",
        colors_js(),
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

    return "\n".join(js)


def build_conditional_css(out_dir: str) -> str:

    search_css = "https://unpkg.com/leaflet-search@2.9.8/dist/leaflet-search.src.css"
    css_string = "   <link rel='stylesheet' href='{}'/>"

    support_dir = os.path.join(os.path.dirname(__file__), "support")
    out_css_dir = os.path.join(out_dir, "css")

    local_css_files = list(
        filter(lambda f: os.path.splitext(f)[1] == ".css", os.listdir(support_dir))
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


def build_conditional_js(marker_file_names: List[str]) -> str:
    search_and_cluster_js = [
        "   <script src='https://unpkg.com/leaflet.markercluster@1.4.1/dist/leaflet.markercluster-src.js' crossorigin=''></script>",
        "   <script src='https://unpkg.com/leaflet-search@2.9.8/dist/leaflet-search.src.js' crossorigin=''></script>",
    ]

    js_string = "   <script src='js/{}'></script>"
    local_js = list(map(lambda s: js_string.format(s), marker_file_names))

    return "\n".join(search_and_cluster_js + local_js)


def build_html(title: str, js: str, extra_js: str, extra_css: str) -> str:
    html = [
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

    return "\n".join(html)
