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
from collections import namedtuple
from logging import Logger, getLogger, info
from time import perf_counter
from typing import Callable

import numpy as np
from kdbush import KDBush

PointCluster = namedtuple("PointCluster", ["x", "y", "zoom", "index", "parent_id",])
Cluster = namedtuple("Cluster", ["x", "y", "zoom", "id", "parent_id", "num_points", "properties"])
Tile = namedtuple("Tile", ["features"], defaults=([],))

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

    # https://github.com/mapbox/supercluster/blob/60d13df9c7d96e9ad16b43c8aca897b5aea38ac9/index.js#L31
    def load(self, points) -> "Supercluster":

        if self.log:
            total_start = perf_counter()

        self.points = points

        clusters = []
        for i in range(len(points)):
            if "geometry" in points[i]:
                clusters.append(self.createPointCluster(points[i], i))

        self.trees[self.max_zoom + 1] = KDBush(
            points=clusters,
            node_size=self.node_size,
            array_dtype=np.float32,
        )

        if self.log:
            print()

        # cluster points on max zoom, then cluster the results on previous zoom, etc.;
        # results in a cluster hierarchy across zoom levels
        for z in range(self.max_zoom, self.min_zoom-1, -1):
            now = perf_counter()

            clusters = self._cluster(clusters, z)
            self.trees[z] = KDBush(
                points=clusters,
                node_size=self.node_size,
                array_type=np.float32
            )

            if self.log:
                print(f"{z}: {len(clusters)} clusters in {perf_counter() - now} seconds")

        if self.log:
            print()

        return self


    # https://github.com/ryanhausen/supercluster/blob/97dbc5687fe0c6c3e63bafc15ad3d942bbd316b6/index.js#L77
    def get_clusters(self, bbox, zoom):

        tree = self.trees[self._limit_zoom(zoom)]

        min_lng, min_lat, max_lng, max_lat = bbox
        ids = tree.range(
            self.lng_x(min_lng),
            self.lat_y(min_lat),
            self.lng_x(max_lng),
            self.lat_y(max_lat),
        )

        clusters = []
        for id in ids:
            c = tree.points[id]
            clusters.append(self.get_cluster_JSON(c) if c.num_points else self.points[c.index])
        return clusters


    # https://github.com/ryanhausen/supercluster/blob/97dbc5687fe0c6c3e63bafc15ad3d942bbd316b6/index.js#L115
    def get_children(self, cluster_id):
        origin_id = self.get_origin_id(cluster_id)
        origin_zoom = self._get_origin_zoom(cluster_id)
        err_msg = "No cluster with the specified id"

        index = self.trees[origin_zoom]
        if not index:
            raise Exception(err_msg)

        origin = index.points[origin_id]
        if not origin:
            raise Exception(err_msg)

        r = self.radius / (self.extent * 2**(origin_zoom -1))
        ids = index.within(origin.x, origin.y, r)
        children = []
        for id in ids:
            c = index.points[id]
            if c.parent_id == cluster_id:
                children.append(self.get_cluster_JSON(c) if c.num_points else self.points[c.index])

        if len(children)==0:
            raise Exception(err_msg)

        return children


    # https://github.com/ryanhausen/supercluster/blob/97dbc5687fe0c6c3e63bafc15ad3d942bbd316b6/index.js#L141
    def get_leaves(self, cluster_id, limit, offset):
        limit = limit if limit else 10
        offset = offset if offset else 0

        leaves = []
        self._append_leaves(leaves, cluster_id, limit, offset, 0)

        return leaves


    # https://github.com/ryanhausen/supercluster/blob/97dbc5687fe0c6c3e63bafc15ad3d942bbd316b6/index.js#L151
    def get_tile(self, z, y, x):
        tree = self.trees[self._limit_zoom(z)]
        z2 = 2**z
        p = self.radius / self.extent
        top = (y - p) / z2
        bottom = (y + 1 + p) / z2

        tile = self._add_tile_features(
            tree.range((x-p) / z2, top, (x + 1 + p) / z2, bottom),
            tree.points,
            x,
            y,
            z2,
            Tile(),
        )

        if x==0:
            self._add_tile_features(
                tree.range(1 - p / z2, top, 1, bottom),
                tree.points,
                z2,
                y,
                z2,
                tile,
            )

        if x == z2 - 1:
            tile = self._add_tile_features(
                tree.range(0, top, p / z2, bottom),
                tree.points,
                -1,
                y,
                z2,
                tile,
            )

        return tile if len(tile.features) else None



    # https://github.com/ryanhausen/supercluster/blob/97dbc5687fe0c6c3e63bafc15ad3d942bbd316b6/index.js#L192
    def _append_leaves(self, result, cluster_id, limit, offset, skipped):
        children = self.get_children(cluster_id)

        for child in children:
            props = child.properties

            if props and "cluster" in props:
                if skipped + props["point_count"] <= offset:
                    skipped += props["point_count"]
                else:
                    skipped = self._append_leaves(
                        result,
                        props["cluster_id"],
                        limit,
                        offset,
                        skipped,
                    )
            elif skipped < offset:
                skipped += 1
            else:
                result.append(child)

            if len(result) == limit:
                break

        return skipped


    # https://github.com/ryanhausen/supercluster/blob/97dbc5687fe0c6c3e63bafc15ad3d942bbd316b6/index.js#L181
    def get_cluster_expansion_zoom(self, cluster_id):
        expansion_zoom = self._get_origin_zoom(cluster_id) - 1
        while expansion_zoom <= self.max_zoom:
            children = self.get_children(cluster_id)
            expansion_zoom += 1
            if len(children) != 1:
                break

            cluster_id = children[0].properties.cluster_id

        return expansion_zoom


    # https://github.com/ryanhausen/supercluster/blob/97dbc5687fe0c6c3e63bafc15ad3d942bbd316b6/index.js#L392
    def get_cluster_JSON(self, cluster):
        return dict(
            type="Feature",
            id=cluster.id,
            properties=self.get_cluster_properties(cluster),
            geometry=dict(type="Point", coordinates=[self.x_lng(cluster.x), self.y_lat(cluster.y)])
        )


    # https://github.com/mapbox/supercluster/blob/60d13df9c7d96e9ad16b43c8aca897b5aea38ac9/index.js#L351
    def create_point_cluster(self, point, id) -> PointCluster:
        x, y = point["geometry"]["coordinates"]
        return PointCluster(x=x, y=y, zoom=np.inf, index=id, parent_id=-1)


    # https://github.com/ryanhausen/supercluster/blob/97dbc5687fe0c6c3e63bafc15ad3d942bbd316b6/index.js#L417
    def get_cluster_properties(self, cluster) -> dict:
        count = cluster.num_points

        if count >= 10000:
            abbrev = f"{round(count / 1000)}k"
        elif count >= 1000:
            abbrev = f"{round(count / 100) / 10}k"
        else:
            abbrev = count

        return dict(
            cluster=True,
            cluster_id = cluster.id,
            point_count = count,
            point_count_abbreviated = abbrev,
            **cluster.properties,
        )


    # https://github.com/ryanhausen/supercluster/blob/97dbc5687fe0c6c3e63bafc15ad3d942bbd316b6/index.js#L220
    def _add_tile_features(self, ids, points, x, y, z2, tile):

        for i in ids:
            c = points[i]
            is_cluster = c.num_points

            if is_cluster:
                tags = self.get_cluster_properties(c)
                px = c.x
                py = c.y
            else:
                p = self.points[c.index]
                tags = p.properties
                px = self.lng_x(p.geometry.coordinates[0])
                py = self.lng_x(p.geometry.coordinates[1])

            f = dict(
                type=1,
                geometry= [[
                    round(self.extent * (px * z2 - x)),
                    round(self.extent * (py * z2 - y)),
                ]],
                **tags
            )

            if is_cluster:
                id = c.id
            elif self.generate_id:
                id = c.index
            elif self.points[c.index].id:
                id = self.points[c.index].id

            if id is not None:
                f.id = id

            tile.features.append(f)

        return tile



    def x_lng(self, x) -> float:
        return x

    def y_lat(self, y) -> float:
        return y

    def lng_x(self, x:float) -> float:
        return x

    def lat_y(self, y:float) -> float:
        return y

    # https://github.com/mapbox/supercluster/blob/60d13df9c7d96e9ad16b43c8aca897b5aea38ac9/index.js#L246
    def _cluster(self, points, zoom:int):
        clusters = []
        r = self.radius / (self.extent * 2**zoom)

        for i in range(len(points)):
            p = points[i]

            if p.zoom <= zoom:
                continue
            p.zoom = zoom

            tree = self.trees[zoom + 1]
            neighbor_ids = tree.within(p.x, p.y, r)

            num_points_origin = max(p.numPoints, 1)
            num_points = num_points_origin

            for neighbor_id in neighbor_ids:
                b = tree.points[neighbor_id]
                if b.zoom > zoom:
                    num_points += max(b.num_points, 1)

            if num_points >= self.min_points:
                wx = p.x * num_points_origin
                wy = p.y * num_points_origin

                if self.reduce_f and num_points_origin > 1:
                    cluster_properties = self._map(p, True)
                else:
                    cluster_properties = None

                # // encode both zoom and point index on which the cluster originated -- offset by total length of features
                id = (i << 5) + (zoom + 1) + len(self.points)

                for neighbor_id in neighbor_ids:
                    b = tree.points[neighbor_id]

                    if b.zoom <= zoom:
                        continue
                    b.zoom = zoom

                    num_points_2 = max(b.num_points, 1)
                    wx = b.x * num_points_origin
                    wy = b.y * num_points_origin

                    b.parentId = id

                    if self.reduce_f:
                        if cluster_properties is None:
                            cluster_properties = self._map(p, True)
                        self.reduce_f(cluster_properties, self.point_reduce_f(b))

                p.parent_id = id
                clusters.append(self.create_cluster(wx / num_points, wy / num_points, id, num_points, cluster_properties))

            else:
                clusters.push(p)

                if num_points > 1:
                    for neighbor_id in neighbor_ids:
                        b = tree.points[neighbor_id]
                        if b.zoom <= zoom:
                            continue
                        clusters.append(b)


    def _map(self, point, clone:bool):
        if point.num_points:
            return dict(**point.properties) if clone else point.properties

        original = self.points[point.index].properties
        result = self.point_reduce_f(original)
        return dict(**result) if clone and result == original else result


    # https://github.com/mapbox/supercluster/blob/60d13df9c7d96e9ad16b43c8aca897b5aea38ac9/index.js#L339
    def create_cluster(self, x, y, id, num_points, properties) -> Cluster:
        return Cluster(
            x=x,
            y=y,
            zoom=np.inf,
            id=id,
            parent_id=-1,
            num_points=num_points,
            properties=properties,
        )



