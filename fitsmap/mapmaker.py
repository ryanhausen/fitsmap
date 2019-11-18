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

import os
import shutil
import sys
from functools import partial, reduce
from itertools import count, repeat
from multiprocessing import Pool
from typing import List, Tuple, Union

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from imageio import imread
from tqdm import tqdm

import fitsmap.convert as convert

# https://github.com/zimeon/iiif/issues/11#issuecomment-131129062
from PIL import Image

Image.MAX_IMAGE_PIXELS = sys.maxsize

Shape = Union[List[int], Tuple[int, int]]

IMG_FORMATS = ["fits", "jpg", "png"]
CAT_FORMAT = ["cat"]
METHOD_RECURSIVE = "recursive"
METHOD_ITERATIVE = "iterative"


def tile_img(
    file_path: str,
    layer_name: str,
    pbar: tqdm,
    tile_size: Shape = [256, 256],
    depth: int = None,
    method: str = METHOD_RECURSIVE,
    image_engine: str = convert.IMG_ENGINE_MPL,
    out_dir: str = None,
) -> Tuple[int, int]:
    file_dir, file_name = os.path.split(file_path)

    fs = file_name.split(".")
    name, ext = "_".join(fs[:-1]), fs[-1]

    root_dir = out_dir if out_dir else file_dir
    tiles_dir = os.path.join(root_dir, name)
    if name not in os.listdir(root_dir):
        os.mkdir(tiles_dir)

    if ext == "fits":
        array = fits.getdata(file_path)
        shape = array.shape
        if len(shape) > 2:
            raise ValueError("FITS files with greater than 2 dims are not supported")
    else:
        array = np.flipud(imread(file_path))

        if len(array.shape) ==3:
            shape = array.shape[:-1]
        elif len(array.shape) == 2:
            shape = array.shape
        else:
            raise ValueError("FitsMap only supports 2D and 3D images.")


    if shape[0] != shape[1]:
        raise ValueError("Only square images are currently supported")

    if depth is None:
        depth = convert._get_depth(array.shape, tile_size)

    convert.array(
        array,
        pbar,
        tile_size=tile_size,
        depth=depth,
        out_dir=tiles_dir,
        method=method,
        image_engine=image_engine,
    )

    return shape


def img_to_layer(
    file_location: str,
    pbar_loc: int,
    tile_size: Shape,
    depth: int = None,
    method: str = METHOD_RECURSIVE,
    image_engine: str = convert.IMG_ENGINE_MPL,
    out_dir: str = None,
) -> Tuple[str, str, int]:

    _, fname = os.path.split(file_location)
    name = os.path.splitext(fname)[0].replace(".", "_").replace("-", "_")

    pbar = tqdm(position=pbar_loc, desc="Converting " + name, unit="tile")

    img_shape = tile_img(
        file_location,
        name,
        pbar,
        tile_size=tile_size,
        depth=depth,
        method=method,
        image_engine=image_engine,
        out_dir=out_dir,
    )

    layer_dir = name + "/{z}/{y}/{x}.png"
    return (layer_dir, name, img_shape)


def dir_to_map(
    directory: str,
    tile_size: Shape = [256, 256],
    depth: int = None,
    method: str = METHOD_RECURSIVE,
    image_engine: str = convert.IMG_ENGINE_MPL,
    title: str = "FitsMap",
    multiprocessing_processes: int = 0,
    cat_wcs_fits_file: str = None,
    out_dir: str = None,
):
    if out_dir:
        _map = _Map(out_dir, title)
    else:
        _map = _Map(directory, title)

    dir_entries = list(
        map(
            lambda d: os.path.join(directory, d),
            filter(
                lambda d: os.path.splitext(d)[1][1:] in IMG_FORMATS,
                os.listdir(directory),
            ),
        )
    )

    kwargs = dict(
        tile_size=tile_size,
        depth=depth,
        method=method,
        image_engine=image_engine,
        out_dir=out_dir,
    )
    layer_func = partial(img_to_layer, **kwargs)

    pbar_loc_generator = count(0)
    if multiprocessing_processes > 0:
        with Pool(multiprocessing_processes) as p:
            built_layers = p.starmap(layer_func, zip(dir_entries, pbar_loc_generator))
    else:
        built_layers = map(layer_func, dir_entries, pbar_loc_generator)

    def update_map(shp1: Tuple[int, int], layer: Tuple[str, str, Shape]) -> np.ndarray:
        p, n, shp2 = layer
        _map.add_tile_layer(p, n)
        return np.maximum(shp1, shp2)

    max_xy = reduce(update_map, built_layers, (0, 0))

    catalogs = list(
        map(
            lambda d: os.path.join(directory, d),
            filter(lambda d: d.endswith(".cat"), os.listdir(directory)),
        )
    )

    if len(catalogs) > 0:
        if cat_wcs_fits_file is None:
            err_msg = [
                "There are catalog(.cat) in files in " + directory + ", but no",
                "value was given for 'cat_wcs_fits_file' skipping catalog coversion",
            ]
            print(" ".join(err_msg))
        else:
            cat_func = partial(convert.catalog, cat_wcs_fits_file, out_dir, max_xy)

            if multiprocessing_processes > 0:
                with Pool(multiprocessing_processes) as p:
                    list(
                        map(
                            _map.add_marker_catalog,
                            list(
                                p.starmap(cat_func, zip(catalogs, pbar_loc_generator))
                            ),
                        )
                    )
            else:
                # unlist this maybe?
                list(
                    map(
                        _map.add_marker_catalog,
                        list(map(cat_func, catalogs, pbar_loc_generator)),
                    )
                )

    _map.build_map()


class _Map:
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

    def add_tile_layer(self, directory, name):
        self.tile_layers.append({"directory": directory, "name": name})

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

    def build_map(self):
        script_text = []

        for tile_layer in self.tile_layers:
            script_text.append(_Map.js_tile_layer(tile_layer, self.max_zoom))

        if self.marker_files:
            script_text.append("\n")
            script_text.append(_Map.js_markers(self.marker_files))

        script_text.append("\n")

        script_text.append(_Map.js_map(self.max_zoom, self.tile_layers))

        if self.marker_files:
            script_text.append("\n")
            script_text.append(_Map.js_marker_search())  # marker searcch

        script_text.append("\n")

        script_text.append(_Map.js_base_layers(self.tile_layers))

        script_text.append(_Map.js_layer_control(self.marker_files))

        script_text = "\n".join(script_text)

        html = self.html_wrapper().replace(_Map.SCRIPT_MARK, script_text)

        with open(os.path.join(self.out_dir, "index.html"), "w") as f:
            f.write(html)

    @staticmethod
    def js_marker_search():
        js = [
            "   var marker_layers = L.layerGroup(markers);",
            "",
            "   function searchHelp(e) {",
            "      map.setView(e.latlng, 8);",
            "      console.log(e.layer)",
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

        cluster_text = "      L.markerClusterGroup({ chunkedLoading: true }),"
        marker_list_text = "      []"

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
                map(lambda s: "      " + s.replace(".cat.js", "") + ",", marker_collections)
            ),
            "   ];",
            "",
            "   var labels = [",
            *list(
                map(lambda s: "      '" + s.replace(".cat.js", "") + "',", marker_collections)
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
            "      center:[-126, 126],",
            "      zoom:0,",
            "      maxNativeZoom:{},".format(max_zoom),
            "      layers:[{}]".format(",".join([t["name"] for t in tile_layers])),
            "   });",
        ]

        return "\n".join(js)

    @staticmethod
    def js_tile_layer(tile_layer, max_zoom):
        js = "   var " + tile_layer["name"]
        js += ' = L.tileLayer("' + tile_layer["directory"] + '"'
        js += ', {attribution:"' + _Map.ATTR + '"});'

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
            _Map.SCRIPT_MARK,
            "   </script>",
            "</body>",
            "</html>",
        ]

        return "\n".join(text)
