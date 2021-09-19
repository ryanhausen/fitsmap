# this is a python port of supercluster.js https://github.com/mapbox/supercluster
#
# released under the following license
# ISC License
#
# Copyright (c) 2021, Mapbox
#
# Permission to use, copy, modify, and/or distribute this software for any purpose
# with or without fee is hereby granted, provided that the above copyright notice
# and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
# TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
# THIS SOFTWARE.
from logging import Logger, getLogger
from time import perf_counter
from typing import Callable

import numpy as np
from kdbush import KDBush


class Supercluster:

    def __init__(
        self,
        min_zoom:int = 0,                       # min zoom to generate clusters on
        max_zoom:int = 16,                      # max zoom level to cluster
        min_points:int = 2,                     # min points to form a cluster
        radus:float = 40,                       # cluster radius in pixels
        extent:int = 256,                       # tile extent
        node_size:int = 64,                     # size fo the kd-tree leaf mode, afftects performance
        log:Logger = getLogger("supercluster"), # whether to log timing info, set to None to disable
        generate_id:bool = False,               # whether to generate numeric ids for input features
        reduce_f:Callable = None,               # a reduce function for calculating custom cluster properties
        point_reduce_f:Callable = lambda i: i,  # properties to use for individual points when running the `reduce_f`
    ):
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.min_points = min_points
        self.radius = radus
        self.extent = extent
        self.node_size = node_size
        self.log = log
        self.generate_id = generate_id
        self.reduce_f = reduce_f
        self.point_reduce_f = point_reduce_f
        self.trees = np.zeros([max_zoom], dtype=object)

    def load(self, points) -> "Supercluster":

        if self.log:
            total_start = perf_counter()

        self.points = points

        clusters = []
        for i in range(len(points)):
            # resume here:
            # https://github.com/mapbox/supercluster/blob/60d13df9c7d96e9ad16b43c8aca897b5aea38ac9/index.js#L44
            pass





        return self

