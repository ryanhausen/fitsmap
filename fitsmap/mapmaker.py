"""
MIT License
Copyright 2018 Ryan Hausen

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import sys

from astropy.io import fits
from imageio import imread

import convert as convert

# https://github.com/zimeon/iiif/issues/11#issuecomment-131129062
from PIL import Image
Image.MAX_IMAGE_PIXELS = sys.maxsize

class MapMaker:
    ACCECPTABLE_FORMATS = [
        'fits',
        'jpg',
        'png'
    ]


    @staticmethod
    def _create_map(file_path,
                   tile_size=[256, 256],
                   depth=None,
                   method='recursive',
                   image_engine='mpl',
                   pixel_match=False):
        file_dir, file_name = os.path.split(file_path)

        fs =  file_name.split('.')
        name, ext = '_'.join(fs[:-1]), fs[-1]

        out_dir = os.path.join(file_dir, name)
        if name not in os.listdir(file_dir):
            os.mkdir(out_dir)

        if ext=='fits':
            array = fits.getdata(file_path)
        else:
            array = imread(file_path)

        if depth is None:
            depth = convert._get_depth(array.shape, tile_size)

        max_zoom = depth

        convert.array(array,
                      tile_size=tile_size,
                      depth=depth,
                      out_dir=out_dir,
                      method=method,
                      image_engine=image_engine)

        return max_zoom

    @staticmethod
    def dir_to_map(directory,
                   tile_size=[256, 256],
                   depth=None,
                   method='recursive',
                   image_engine='mpl',
                   title='FitsMap',
                   pixel_match=False):

        _map = _Map(directory, title)

        # 18 is the default max zoom for leaflet
        max_zoom = 1000
        for f in os.listdir(directory):
            fs =  f.split('.')
            name, ext = '.'.join(fs[:-1]), fs[-1]
            name = name.replace('.', '_').replace('-', '_')


            if ext in MapMaker.ACCECPTABLE_FORMATS:
                mz = MapMaker._create_map(os.path.join(directory, f),
                                          tile_size=tile_size,
                                          depth=depth,
                                          method=method,
                                          image_engine=image_engine)
                img_directory = name + '/{z}/{y}/{x}.png'
                _map.add_tile_layer(img_directory, name)
                max_zoom = min(max_zoom, mz)

        _map.max_zoom = max_zoom
        _map.build_map()


class _Map:
    SCRIPT_MARK = '!!!FITSMAP!!!'
    ATTR = "<a href=''>FitsMap</a>"

    def __init__(self, out_dir, title):
        self.out_dir = out_dir
        self.title=title
        self.tile_layers = []
        self.min_zoom = 0
        self.max_zoom = 0
        self.var_map = {'center':None, 'zoom':0, 'layers':[]}
        self.var_overlays = {}


    def add_tile_layer(self, directory, name):
        self.tile_layers.append({'directory':directory, 'name':name})

    def build_map(self):
        script_text = []

        for tile_layer in self.tile_layers:
            script_text.append(_Map.js_tile_layer(tile_layer))

        script_text.append('\n')

        script_text.append(_Map.js_map(self.max_zoom, self.tile_layers))

        script_text.append('\n')

        script_text.append(_Map.js_base_layers(self.tile_layers))

        script_text.append(_Map.js_layer_control())

        script_text = '\n'.join(script_text)

        html = self.make_header().replace(_Map.SCRIPT_MARK, script_text)

        with open(os.path.join(self.out_dir, 'index.html'), 'w') as f:
            f.write(html)


    @staticmethod
    def js_map(max_zoom, tile_layers):
        js = [
            'var map = L.map("map", {',
            '   crs: L.CRS.Simple,',
            '   center:[0, 0],',
            '   zoom:0,',
            '   minZoom:0,',
            '   maxZoom:10',
            '   maxNativeZoom:{},'.format(max_zoom),
            '   layers:[{}]'.format(','.join([t['name'] for t in tile_layers])),
            '});'
        ]

        return '\n'.join(js)

    @staticmethod
    def js_tile_layer(tile_layer):
        js = 'var ' + tile_layer['name']
        js += ' = L.tileLayer("' + tile_layer['directory'] + '"'
        js += ', {attribution:"' + _Map.ATTR + '"});'

        return js

    @staticmethod
    def js_base_layers(tile_layers):
        js = ['var baseLayers = {']
        js.extend('"{0}": {0},'.format(t['name']) for t in tile_layers)
        js.append('};')

        return '\n'.join(js)

    @staticmethod
    def js_layer_control():
        return 'L.control.layers(baseLayers, {}).addTo(map);'

    def make_header(self):
        text =  [
            '<!DOCTYPE html>',
            '<html>',
            '<head>',
            '   <title>{}</title>'.format(self.title),
            '   <meta charset="utf-8" />',
	        '   <meta name="viewport" content="width=device-width, initial-scale=1.0">',
	        '   <link rel="shortcut icon" type="image/x-icon" href="docs/images/favicon.ico" />',
            '   <link rel="stylesheet" href="https://unpkg.com/leaflet@1.3.4/dist/leaflet.css" integrity="sha512-puBpdR0798OZvTTbP4A8Ix/l+A4dHDD0DGqYW6RQ+9jxkRFclaxxQb/SJAWZfWAkuyeQUytO7+7N4QKrDh+drA==" crossorigin=""/>',
            '   <script src="https://unpkg.com/leaflet@1.3.4/dist/leaflet.js" integrity="sha512-nMMmRyTVoLYqjP9hrbed9S+FzjZHW5gY1TWCHA5ckwXZBadntCNs8kEqAWdrb9O7rxbCaA4lKTIWjDXZxflOcA==" crossorigin=""></script>',
	        '   <style>',
		    '       html, body {',
			'       height: 100%;',
			'       margin: 0;',
            '       }',
		    '       #map {',
            '           width: 100%;',
			'           height: 100%;',
		    '       }',
	        '   </style>',
            '</head>',
            '<body>',
            '   <div id="map"></div>',
	        '   <script>',
            _Map.SCRIPT_MARK,
	        '   </script>',
            '</body>',
            '</html>'
        ]

        return '\n'.join(text)