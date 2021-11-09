# this is a python port of supercluster.js
# https://github.com/mapbox/supercluster
#
# released under the following license ISC License
#
# Copyright (c) 2021, Mapbox
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.
import math
from typing import Callable, Tuple
import numpy as np
from fitsmap.kdbush import KDBush
from tqdm import tqdm


def default_map(i):
    return i


def default_udpate_f():
    pass


def default_get_x(p):
    return p["x"]


def default_get_y(p):
    return p["y"]


class Supercluster:
    def __init__(
        self,
        min_zoom: int = 0,  # min zoom to generate clusters on
        max_zoom: int = 16,  # max zoom level to cluster
        min_points: int = 2,  # min points to form a cluster
        radius: float = 40,  # cluster radius in pixels
        extent: int = 512,  # tile extent
        node_size: int = 64,  # size for the kd-tree leaf mode, afftects performance
        log: bool = False,  # whether to log timing info, set to None to disable
        generate_id: bool = False,  # whether to generate numeric ids for input features
        reduce: Callable = None,  # a reduce function for calculating custom cluster properties
        map: Callable = default_map,  # properties to use for individual points when running the `reduce`
        alternate_CRS: Tuple[
            int, int
        ] = (),  # if using a simple CRS set the tuple to [max_x, max_y]
        update_f: Callable = default_udpate_f,
    ):
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.min_points = min_points
        self.radius = radius
        self.extent = extent
        self.node_size = node_size
        self.log = log
        self.generate_id = generate_id
        self.reduce = reduce
        self.map = map
        # length of +2 to account for zoom=0 and 1 past max zoom where clustering
        # doesn't happen
        self.trees = np.zeros([max_zoom + 2], dtype=object)
        self.alternate_CRS = alternate_CRS
        self.update_f = update_f

    # https://github.com/mapbox/supercluster/blob/60d13df9c7d96e9ad16b43c8aca897b5aea38ac9/index.js#L31
    def load(self, points) -> "Supercluster":

        self.points = points

        clusters = []
        for i in range(len(points)):
            if points[i].get("geometry", []):
                clusters.append(self.create_point_cluster(points[i], i))

        if self.log:
            self.update_f()

        self.trees[self.max_zoom + 1] = KDBush(
            points=clusters,
            node_size=self.node_size,
            array_dtype=np.float32,
            get_x=default_get_x,
            get_y=default_get_y,
        )

        if self.log:
            self.update_f()

        # cluster points on max zoom, then cluster the results on previous zoom, etc.;
        # results in a cluster hierarchy across zoom levels
        for z in range(self.max_zoom, self.min_zoom - 1, -1):

            clusters = self._cluster(clusters, z)
            self.trees[z] = KDBush(
                points=clusters,
                node_size=self.node_size,
                array_dtype=np.float32,
                get_x=default_get_x,
                get_y=default_get_y,
            )

            if self.log:
                self.update_f()

        return self

    # https://github.com/ryanhausen/supercluster/blob/97dbc5687fe0c6c3e63bafc15ad3d942bbd316b6/index.js#L77
    def get_clusters(self, bbox, zoom):

        if self.alternate_CRS:
            min_lng, min_lat, max_lng, max_lat = bbox
        else:
            min_lng = ((bbox[0] + 180) % 360 + 360) % 360 - 180
            min_lat = max(-90, min(90, bbox[1]))
            max_lng = (
                180 if bbox[2] == 180 else ((bbox[2] + 180) % 360 + 360) % 360 - 180
            )
            max_lat = max(-90, min(90, bbox[3]))

            if bbox[2] - bbox[0] >= 360:
                min_lng = -180
                max_lng = 180
            elif min_lng > max_lng:
                easternHem = self.get_clusters([min_lng, min_lat, 180, max_lat], zoom)
                westernHem = self.get_clusters([-180, min_lat, max_lng, max_lat], zoom)
                return easternHem + westernHem

        tree = self.trees[self._limit_zoom(zoom)]

        if self.alternate_CRS:
            ids = tree.range(
                self.lng_x(min_lng),
                self.lat_y(min_lat),
                self.lng_x(max_lng),
                self.lat_y(max_lat),
            )
        else:
            ids = tree.range(
                self.lng_x(min_lng),
                self.lat_y(max_lat),
                self.lng_x(max_lng),
                self.lat_y(min_lat),
            )

        clusters = []
        for id in ids:
            c = tree.points[id]
            clusters.append(
                self.get_cluster_JSON(c)
                if c.get("num_points", 0)
                else self.points[c["index"]]
            )
        return clusters

    # https://github.com/ryanhausen/supercluster/blob/97dbc5687fe0c6c3e63bafc15ad3d942bbd316b6/index.js#L115
    def get_children(self, cluster_id):
        origin_id = self._get_origin_id(cluster_id)
        origin_zoom = self._get_origin_zoom(cluster_id)
        err_msg = "No cluster with the specified id"

        index = self.trees[origin_zoom]
        if not index:
            raise Exception(err_msg)

        origin = index.points[origin_id]
        if not origin:
            raise Exception(err_msg)

        r = self.radius / (self.extent * 2 ** (origin_zoom - 1))
        ids = index.within(origin["x"], origin["y"], r)
        children = []
        for id in ids:
            c = index.points[id]
            if c["parent_id"] == cluster_id:
                children.append(
                    self.get_cluster_JSON(c)
                    if "num_points" in c
                    else self.points[c["index"]]
                )

        if len(children) == 0:
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
    def get_tile(self, z, x, y):
        tree = self.trees[self._limit_zoom(z)]
        z2 = 2 ** z
        p = self.radius / self.extent
        top = (y - p) / z2
        bottom = (y + 1 + p) / z2

        tile = self._add_tile_features(
            tree.range((x - p) / z2, top, (x + 1 + p) / z2, bottom),
            tree.points,
            x,
            y,
            z2,
            dict(features=[]),
        )

        if x == 0:
            tile = self._add_tile_features(
                tree.range(1 - p / z2, top, 1, bottom), tree.points, z2, y, z2, tile,
            )

        if x == z2 - 1:
            tile = self._add_tile_features(
                tree.range(0, top, p / z2, bottom), tree.points, -1, y, z2, tile,
            )

        return tile if len(tile["features"]) else None

    # https://github.com/ryanhausen/supercluster/blob/97dbc5687fe0c6c3e63bafc15ad3d942bbd316b6/index.js#L181
    def get_cluster_expansion_zoom(self, cluster_id):
        expansion_zoom = self._get_origin_zoom(cluster_id) - 1

        while expansion_zoom <= self.max_zoom:
            children = self.get_children(cluster_id)
            expansion_zoom += 1
            if len(children) != 1:
                break

            cluster_id = children[0]["properties"]["cluster_id"]

        return expansion_zoom

    # https://github.com/ryanhausen/supercluster/blob/97dbc5687fe0c6c3e63bafc15ad3d942bbd316b6/index.js#L192
    def _append_leaves(self, result, cluster_id, limit, offset, skipped):
        children = self.get_children(cluster_id)

        for child in children:
            props = child["properties"]

            if props and "cluster" in props:
                if skipped + props["point_count"] <= offset:
                    skipped += props["point_count"]
                else:
                    skipped = self._append_leaves(
                        result, props["cluster_id"], limit, offset, skipped,
                    )
            elif skipped < offset:
                skipped += 1
            else:
                result.append(child)

            if len(result) == limit:
                break

        return skipped

    # https://github.com/ryanhausen/supercluster/blob/97dbc5687fe0c6c3e63bafc15ad3d942bbd316b6/index.js#L220
    def _add_tile_features(self, ids, points, x, y, z2, tile):

        for i in ids:
            c = points[i]
            is_cluster = "num_points" in c

            if is_cluster:
                tags = self.get_cluster_properties(c)
                px = self.x_lng(c["x"])
                py = self.y_lat(c["y"])
            else:
                p = self.points[c["index"]]
                tags = p.get("tags", None)
                px = p["geometry"]["coordinates"][0]
                py = p["geometry"]["coordinates"][1]

            f = dict(
                type=1,
                geometry=[
                    round(self.extent * (px * z2 - x)),
                    round(self.extent * (py * z2 - y)),
                ],
                properties={"global_x": px, "global_y": py, **tags},
            )

            if is_cluster:
                id = c["id"]
            elif self.generate_id:
                id = c["index"]
            elif "id" in self.points[c["index"]]:
                id = self.points[c["index"]]["id"]
            else:
                id = None

            if id is not None:
                f["id"] = id

            tile["features"].append(f)

        return tile

    # https://github.com/ryanhausen/supercluster/blob/97dbc5687fe0c6c3e63bafc15ad3d942bbd316b6/index.js#L264
    def _limit_zoom(self, z):
        return max(self.min_zoom, min(z, self.max_zoom + 1))

    # https://github.com/mapbox/supercluster/blob/60d13df9c7d96e9ad16b43c8aca897b5aea38ac9/index.js#L246
    def _cluster(self, points, zoom: int):
        clusters = []
        r = self.radius / (self.extent * 2 ** zoom)

        for i in range(len(points)):
            p = points[i]

            if p["zoom"] <= zoom:
                continue
            p["zoom"] = zoom

            tree = self.trees[zoom + 1]
            neighbor_ids = tree.within(p["x"], p["y"], r)

            num_points_origin = p.get("num_points", 1)
            num_points = num_points_origin

            for neighbor_id in neighbor_ids:
                b = tree.points[neighbor_id]
                if b["zoom"] > zoom:
                    num_points += b.get("num_points", 1)

            if num_points >= self.min_points:
                wx = p["x"] * num_points_origin
                wy = p["y"] * num_points_origin

                if self.reduce and num_points_origin > 1:
                    cluster_properties = self._map(p, True)
                else:
                    cluster_properties = None

                # // encode both zoom and point index on which the cluster originated -- offset by total length of features
                id = (i << 5) + (zoom + 1) + len(self.points)

                for neighbor_id in neighbor_ids:
                    b = tree.points[neighbor_id]

                    if b["zoom"] <= zoom:
                        continue
                    b["zoom"] = zoom

                    num_points_2 = b.get("num_points", 1)
                    wx += b["x"] * num_points_2
                    wy += b["y"] * num_points_2

                    b["parent_id"] = id

                    if self.reduce:
                        if cluster_properties is None:
                            cluster_properties = self._map(p, True)
                        cluster_properties = self.reduce(
                            cluster_properties, self._map(b)
                        )

                p["parent_id"] = id
                clusters.append(
                    self.create_cluster(
                        wx / num_points,
                        wy / num_points,
                        id,
                        num_points,
                        cluster_properties,
                    )
                )

            else:
                clusters.append(p)

                if num_points > 1:
                    for neighbor_id in neighbor_ids:
                        b = tree.points[neighbor_id]
                        if b["zoom"] <= zoom:
                            continue
                        b["zoom"] = zoom
                        clusters.append(b)

        return clusters

    # get index of the point from which the cluster originated
    # https://github.com/mapbox/supercluster/blob/60d13df9c7d96e9ad16b43c8aca897b5aea38ac9/index.js#L320
    def _get_origin_id(self, clusterId):
        return (clusterId - len(self.points)) >> 5

    # https://github.com/mapbox/supercluster/blob/60d13df9c7d96e9ad16b43c8aca897b5aea38ac9/index.js#L325
    def _get_origin_zoom(self, clusterId):
        return (clusterId - len(self.points)) % 32

    # https://github.com/mapbox/supercluster/blob/60d13df9c7d96e9ad16b43c8aca897b5aea38ac9/index.js#L329
    def _map(self, point, clone: bool = False):
        if "num_points" in point:
            return dict(**point["properties"]) if clone else point["properties"]

        original = self.points[point["index"]]["properties"]
        result = self.map(original)
        return dict(**result) if clone and result == original else result

    # https://github.com/mapbox/supercluster/blob/60d13df9c7d96e9ad16b43c8aca897b5aea38ac9/index.js#L339
    def create_cluster(self, x, y, id, num_points, properties):
        return dict(
            x=np.asarray(x, dtype=np.float32),
            y=np.asarray(y, dtype=np.float32),
            zoom=np.inf,
            id=id,
            parent_id=-1,
            num_points=num_points,
            properties=properties,
        )

    # https://github.com/mapbox/supercluster/blob/60d13df9c7d96e9ad16b43c8aca897b5aea38ac9/index.js#L351
    def create_point_cluster(self, p, id):
        x, y = p["geometry"]["coordinates"]
        return dict(
            x=np.asarray(
                self.lng_x(x), dtype=np.float32
            ),  # projected point coordinates
            y=np.asarray(self.lat_y(y), dtype=np.float32),
            zoom=np.inf,  # the last zoom the point was processed at
            index=id,  # index of the source feature in the original input array,
            parentId=-1,  # parent cluster id
            tags=p["tags"],  # additional properties not needed for clustering
        )

    # https://github.com/mapbox/supercluster/blob/60d13df9c7d96e9ad16b43c8aca897b5aea38ac9/index.js#L362
    def get_cluster_JSON(self, cluster):
        return dict(
            type="Feature",
            id=cluster["id"],
            properties=self.get_cluster_properties(cluster),
            geometry=dict(
                type="Point",
                coordinates=[self.x_lng(cluster["x"]), self.y_lat(cluster["y"])],
            ),
        )

    def get_cluster_properties(self, cluster):
        count = cluster["num_points"]

        if count >= 1000000:
            abbrev = f"{round(count / 1000000)}M"
        elif count >= 10000:
            abbrev = f"{round(count / 1000)}k"
        elif count > 1000:
            abbrev = f"{round(count / 100  / 10)}k"
        else:
            abbrev = count

        props = cluster["properties"] if cluster["properties"] else {}
        return dict(
            cluster=True,
            cluster_id=cluster["id"],
            point_count=count,
            point_count_abbreviated=abbrev,
            **props,
        )

    # needs to bound lng to [0..1]
    def lng_x(self, lng):
        if self.alternate_CRS:
            return lng / self.alternate_CRS[0]
        else:
            return lng / 360 + 0.5

    # needs to bound lat to [0..1]
    def lat_y(self, lat):
        if self.alternate_CRS:
            return lat / self.alternate_CRS[1]
        else:
            sin = math.sin(lat * math.pi / 180)
            if sin in [-1, 1]:
                y = -sin
            else:
                y = 0.5 - 0.25 * math.log((1 + sin) / (1 - sin)) / math.pi
            return min(max(y, 0), 1)

    # return to original coordinate system
    def x_lng(self, x):
        if self.alternate_CRS:
            return x * self.alternate_CRS[0]
        else:
            return (x - 0.5) * 360

    # return to original coordinate system
    def y_lat(self, y):
        if self.alternate_CRS:
            return y * self.alternate_CRS[1]
        else:
            y2 = (180 - y * 360) * math.pi / 180
            return 360 * math.atan(math.exp(y2)) / math.pi - 90
