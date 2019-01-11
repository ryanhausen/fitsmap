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

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def _dynamically_range(arr, scale_max):
    # if (arr < 0).any():
    #     arr += abs(arr.min())

    # arr = arr / scale_max
    # arr = (arr * 255).astype(np.int32)

    # return arr
    pcmin = 1e-4
    dynrange = 1000
    arr = np.clip(arr, pcmin, pcmin*dynrange)

    num = (np.log10(arr) - np.log10(pcmin))
    denom = (np.log10(pcmin*dynrange) - np.log10(pcmin))
    arr = num / denom
    arr = 0.5 * arr + 0.5

    return arr

# TODO: Try to understand why leaflet prefers the coords like this
def _get_new_coords(y, x):
    adj = lambda z: 2*z
    adj_y = adj(y)
    adj_x = adj(x)

    return [ (adj_y+1, adj_x),
             (adj_y+1, adj_x+1),
             (adj_y, adj_x),
             (adj_y, adj_x+1)]

def _get_depth(shape, tile_size):
    return int(np.log2(shape[0] / tile_size[0]))

def _build_path(depth, y, x, out_dir):
    depth, y, x = str(depth), str(y), str(x)

    z_dir = os.path.join(out_dir, depth)

    if depth not in os.listdir(out_dir):
        os.mkdir(z_dir)

    y_dir = os.path.join(z_dir, y)
    if y not in os.listdir(z_dir):
        os.mkdir(y_dir)

    img_path = os.path.join(y_dir, '{}.png'.format(x))

    return img_path

def _convert_and_save(array,
                      depth,
                      y,
                      x,
                      out_dir,
                      tile_size,
                      image_engine,
                      vmax):
    path = _build_path(depth, y, x, out_dir)

    if image_engine=='PIL':
        img = Image.fromarray(array)
        img.thumbnail(tile_size)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img.save(path, 'PNG')
        del img
    else:
        f = plt.figure(dpi=100)
        f.set_size_inches([tile_size[0]/100, tile_size[1]/100])
        plt.imshow(array, origin='lower', cmap='gray', vmin=0, vmax=vmax)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis('off')
        plt.savefig(path, dpi=100, bbox_inches=0)
        plt.close(f)

def array(array,
          tile_size=[256, 256],
          depth=None,
          out_dir='.',
          method='recursive',
          image_engine='mpl'):
    if method == 'recursive':
        if depth is None:
            depth = _get_depth(array.shape, tile_size)

        if method=='recursive':
            _build_recursively(array,
                            (0,0),
                            0,
                            depth,
                            out_dir,
                            tile_size,
                            image_engine,
                            array.max())
        elif method=='iterative':
            raise NotImplementedError('iterative not supported')
        else:
            raise ValueError('{} invalid. Please use recursive'.format(method))


def _build_recursively(array,
                       coords,
                       depth,
                       goal,
                       out_dir,
                       tile_size,
                       image_engine,
                       vmax):
    y, x = coords


    _convert_and_save(array,
                      depth,
                      y,
                      x,
                      out_dir,
                      tile_size,
                      image_engine,
                      array.max())

    if depth < goal:
        ax0, ax1 = array.shape[0], array.shape[1]

        split_0 = ax0 // 2
        split_1 = ax1 // 2


        slices = [
            (slice(0, split_0), slice(0, split_1)),
            (slice(0, split_0), slice(split_1, ax1)),
            (slice(split_0, ax0), slice(0, split_1)),
            (slice(split_0, ax0), slice(split_1, ax1))
        ]

        for (_ys, _xs), crds in zip(slices, _get_new_coords(y, x)):
            print(depth+1, _ys, _xs, crds)
            arr = array[_ys, _xs]
            _build_recursively(arr,
                               crds,
                               depth+1,
                               goal,
                               out_dir,
                               tile_size,
                               image_engine,
                               vmax)
            del arr




