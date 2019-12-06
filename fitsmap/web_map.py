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

"""Helper class for generating a web page with all the JS needed to present the map."""

import os
import shutil
from itertools import repeat
from typing import List


class Map:
    """A helper class for tracking map elements and support file locations.

    Designed for internal use. Any method/variable can be deprecated or changed
    without consideration.
    """

    SCRIPT_MARK = "!!!FITSMAP!!!"
    ATTR = "<a href='https://github.com/ryanhausen/fitsmap'>FitsMap</a>"

    def __init__(self, out_dir, title):
        self.out_dir = out_dir
        self.title = title
        self.tile_layers = []
        self.marker_files = []
        self.min_zoom = 0
        self.max_zoom = 0
        self.var_map = {"center": None, "zoom": 0, "layers": []}
        self.var_overlays = {}

    def add_tile_layer(self, name: str):
        zooms = list(map(int, os.listdir(os.path.join(self.out_dir, name))))

        self.tile_layers.append(
            dict(
                directory=name + "/{z}/{y}/{x}.png",
                name=name,
                min_zoom=min(zooms),
                max_native_zoom=max(zooms),
            )
        )

    def add_marker_catalog(self, json_file: str):
        self.marker_files.append(json_file)

    def get_conditional_css(self):
        css_files = []
        if self.marker_files:
            support_dir = os.path.join(os.path.dirname(__file__), "support")

            css_files.append(
                "https://unpkg.com/leaflet-search@2.9.8/dist/leaflet-search.src.css"
            )
            for f in os.listdir(support_dir):
                _, ext = os.path.splitext(f)
                if ext != ".css":
                    continue

                src = os.path.join(support_dir, f)

                dst = os.path.join(self.out_dir, ext[1:])
                if ext[1:] not in os.listdir(self.out_dir):
                    os.mkdir(dst)

                css_files.append("css/" + f)
                shutil.copy2(src, dst)

        css_string = "   <link rel='stylesheet' href='{}'/>"

        if css_files:
            return list(map(lambda x: css_string.format(x), css_files))
        else:
            return [""]

    def get_marker_src_entries(self):
        if self.marker_files:
            search_and_cluster_js = [
                "   <script src='https://unpkg.com/leaflet.markercluster@1.4.1/dist/leaflet.markercluster-src.js' crossorigin=''></script>",
                "   <script src='https://unpkg.com/leaflet-search@2.9.8/dist/leaflet-search.src.js' crossorigin=''></script>",
            ]

            js_string = "   <script src='js/{}'></script>"
            return search_and_cluster_js + list(
                map(lambda x: js_string.format(x), self.marker_files)
            )
        else:
            return [""]

    def finalize_max_zoom(self):
        deepest_zoom = max(t["max_native_zoom"] for t in self.tile_layers)

        any(map(lambda t: t.update(dict(max_zoom=deepest_zoom + 5)), self.tile_layers))

    def build_map(self):

        script_text = []

        self.finalize_max_zoom()
        for tile_layer in self.tile_layers:
            script_text.append(Map.js_tile_layer(tile_layer))

        if self.marker_files:
            script_text.append("\n")
            script_text.append(Map.js_markers(self.marker_files))

        script_text.append("\n")

        script_text.append(Map.js_map(self.max_zoom, self.tile_layers))

        if self.marker_files:
            script_text.append("\n")
            script_text.append(Map.js_marker_search())  # marker searcch

        script_text.append("\n")

        script_text.append(Map.js_base_layers(self.tile_layers))

        script_text.append(Map.js_layer_control(self.marker_files))

        script_text = "\n".join(script_text)

        html = self.html_wrapper().replace(Map.SCRIPT_MARK, script_text)

        with open(os.path.join(self.out_dir, "index.html"), "w") as f:
            f.write(html)

    @staticmethod
    def js_marker_search():
        js = [
            "   var marker_layers = L.layerGroup(markers);",
            "",
            "   function searchHelp(e) {",
            "      map.setView(e.latlng, 8);",
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

        return "\n".join(js)

    @staticmethod
    def js_markers(marker_collections: List[str]):

        cluster_text = "      L.markerClusterGroup({ }),"
        marker_list_text = "      [],"

        js = [
            "   var markers = [",
            *list(repeat(cluster_text, len(marker_collections))),
            "   ];",
            "",
            "   var markerList = [",
            *list(repeat(marker_list_text, len(marker_collections))),
            "   ];",
            "",
            "   var collections = [",
            *list(
                map(
                    lambda s: "      " + s.replace(".cat.js", "") + ",",
                    marker_collections,
                )
            ),
            "   ];",
            "",
            "   var labels = [",
            *list(
                map(
                    lambda s: "      '" + s.replace(".cat.js", "") + "',",
                    marker_collections,
                )
            ),
            "   ];",
            "",
            "   for (i = 0; i < collections.length; i++){",
            "      collection = collections[i];",
            "",
            "      for (j = 0; j < collection.length; j++){",
            "         src = collection[j];",
            "",
            "         markerList[i].push(L.circleMarker([src.y, src.x], {",
            "            catalog_id: src.catalog_id",
            "         }).bindPopup(src.desc))",
            "      }",
            "   }",
            "",
            "   for (i = 0; i < collections.length; i++){",
            "      markers[i].addLayers(markerList[i]);",
            "   }",
        ]

        return "\n".join(js)

    @staticmethod
    def js_map(max_zoom, tile_layers):
        js = [
            '   var map = L.map("map", {',
            "      crs: L.CRS.Simple,",
            "      zoom: " + str(max(map(lambda t: t["min_zoom"], tile_layers))) + ",",
            "      minZoom: "
            + str(max(map(lambda t: t["min_zoom"], tile_layers)))
            + ",",
            "      center:[-126, 126],",
            "      layers:[{}]".format(",".join([t["name"] for t in tile_layers])),
            "   });",
        ]

        return "\n".join(js)

    @staticmethod
    def js_tile_layer(tile_layer):
        js = "   var " + tile_layer["name"]
        js += ' = L.tileLayer("' + tile_layer["directory"] + '"'
        js += ", { "
        js += 'attribution:"' + Map.ATTR + '",'
        js += "minZoom: " + str(tile_layer["min_zoom"]) + ","
        js += "maxZoom: " + str(tile_layer["max_zoom"]) + ","
        js += "maxNativeZoom: " + str(tile_layer["max_native_zoom"]) + ","
        js += "});"

        return js

    @staticmethod
    def js_base_layers(tile_layers):
        js = ["   var baseLayers = {"]
        js.extend('      "{0}": {0},'.format(t["name"]) for t in tile_layers)
        js.append("   };")

        return "\n".join(js)

    @staticmethod
    def js_layer_control(markers_files: List[str]):
        if markers_files:
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

    def html_wrapper(self):
        text = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "   <title>{}</title>".format(self.title),
            '   <meta charset="utf-8" />',
            '   <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            '   <link rel="shortcut icon" type="image/x-icon" href="docs/images/favicon.ico" />',
            '   <link rel="stylesheet" href="https://unpkg.com/leaflet@1.3.4/dist/leaflet.css" integrity="sha512-puBpdR0798OZvTTbP4A8Ix/l+A4dHDD0DGqYW6RQ+9jxkRFclaxxQb/SJAWZfWAkuyeQUytO7+7N4QKrDh+drA==" crossorigin=""/>',
            *self.get_conditional_css(),
            "   <script src='https://unpkg.com/leaflet@1.3.4/dist/leaflet.js' integrity='sha512-nMMmRyTVoLYqjP9hrbed9S+FzjZHW5gY1TWCHA5ckwXZBadntCNs8kEqAWdrb9O7rxbCaA4lKTIWjDXZxflOcA==' crossorigin=''></script>",
            *self.get_marker_src_entries(),
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
            Map.SCRIPT_MARK,
            "   </script>",
            "</body>",
            "</html>",
        ]

        return "\n".join(text)
