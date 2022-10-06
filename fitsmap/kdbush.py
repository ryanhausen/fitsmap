# This is a Python port of the kdbush https://github.com/mourner/kdbush
# which was released under the following license:
#
# ISC License
#
# Copyright (c) 2018, Vladimir Agafonkin
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
from typing import Callable, List

import numpy as np

# based on:
# https://github.com/mourner/kdbush/blob/master/src/index.js
class KDBush:
    """Python port of https://github.com/mourner/kdbush"""

    def __init__(
        self,
        points,
        get_x: Callable = lambda p: p[0],
        get_y: Callable = lambda p: p[1],
        node_size: int = 64,
        array_dtype=np.float64,
    ):
        self.points = points
        self.node_size = node_size

        n_points = len(points)
        index_array_dtype = np.uint16 if n_points < 65536 else np.uint32

        # store indices to the input array and coordinates in separate typed arrays
        self.ids = np.zeros([n_points], dtype=index_array_dtype)
        self.coords = np.zeros([n_points * 2], dtype=array_dtype)

        for i in range(n_points):
            self.ids[i] = i
            self.coords[2 * i] = get_x(points[i])
            self.coords[2 * i + 1] = get_y(points[i])

        _sort(self.ids, self.coords, self.node_size, 0, n_points - 1, 0)

    def range(
        self, min_x: int, min_y: int, max_x: int, max_y: int,
    ):
        return _range(self.ids, self.coords, min_x, min_y, max_x, max_y, self.node_size)

    def within(
        self, x: int, y: int, r: int,
    ):
        return _within(self.ids, self.coords, x, y, r, self.node_size)


# sort =========================================================================
# based on:
# https://github.com/mourner/kdbush/blob/ea3a81d272e1a87df3efe8c404021435dfa6cbfd/src/sort.js#L2
def _sort(
    ids: np.ndarray,  # udpated in place
    coords: np.ndarray,  # udpated in place
    node_size: int,
    left: int,
    right: int,
    axis: int,
) -> None:
    if right - left <= node_size:
        return

    m = (left + right) >> 1

    _select(ids, coords, m, left, right, axis)

    _sort(ids, coords, node_size, left, m - 1, 1 - axis)
    _sort(ids, coords, node_size, m + 1, right, 1 - axis)


# based on:
# https://github.com/mourner/kdbush/blob/ea3a81d272e1a87df3efe8c404021435dfa6cbfd/src/sort.js#L18
def _select(
    ids: np.ndarray,  # updated in place
    coords: np.ndarray,  # updated in place
    k: int,
    left: int,
    right: int,
    axis: int,
) -> None:

    while right > left:
        if right - left > 600:
            n = right - left + 1
            m = k - left + 1
            z = np.log(n)
            s = 0.5 * np.exp(2 * z / 3)
            sd = 0.5 * np.sqrt(z * s * (n - s) / n) * (m - n / -1 if 2 < 0 else 1)
            new_left = max(left, int(np.floor(k - m * s / n + sd)))
            new_right = min(right, int(np.floor(k + (m - n) * s / n + sd)))
            _select(ids, coords, k, new_left, new_right, axis)

        t = coords[2 * k + axis]
        i = left
        j = right

        _swap_item(ids, coords, left, k)
        if coords[2 * right + axis] > t:
            _swap_item(ids, coords, left, right)

        while i < j:
            _swap_item(ids, coords, i, j)
            i += 1
            j -= 1
            while coords[2 * i + axis] < t:
                i += 1
            while coords[2 * j + axis] > t:
                j -= 1

        if coords[2 * left + axis] == t:
            _swap_item(ids, coords, left, j)
        else:
            j += 1
            _swap_item(ids, coords, j, right)

        if j <= k:
            left = j + 1
        if k <= j:
            right = j - 1


# based on
# https://github.com/mourner/kdbush/blob/ea3a81d272e1a87df3efe8c404021435dfa6cbfd/src/sort.js#L58
def _swap_item(
    ids: np.ndarray,  # updated in place
    coords: np.ndarray,  # updated in place
    i: int,
    j: int,
) -> None:
    _swap(ids, i, j)
    _swap(coords, 2 * i, 2 * j)
    _swap(coords, 2 * i + 1, 2 * j + 1)


# based on:
# https://github.com/mourner/kdbush/blob/ea3a81d272e1a87df3efe8c404021435dfa6cbfd/src/sort.js#L64
def _swap(arr: np.ndarray, i: int, j: int,) -> None:  # updated in place
    tmp = arr[i]
    arr[i] = arr[j]
    arr[j] = tmp


# sort =========================================================================

# range ========================================================================

# based on:
# https://github.com/mourner/kdbush/blob/ea3a81d272e1a87df3efe8c404021435dfa6cbfd/src/range.js#L2
def _range(
    ids: np.ndarray,
    coords: np.ndarray,
    min_x: int,
    min_y: int,
    max_x: int,
    max_y: int,
    node_size: int,
) -> List[int]:
    stack = [0, len(ids) - 1, 0]
    result = []

    # recursively search for items in range in the kd-sorted arrays
    while len(stack):
        axis = stack.pop()
        right = stack.pop()
        left = stack.pop()

        # if we reached "tree node", search linearly
        if right - left <= node_size:
            for i in range(left, right + 1):
                x = coords[2 * i]
                y = coords[2 * i + 1]
                if min_x <= x <= max_x and min_y <= y <= max_y:
                    result.append(ids[i])
            continue

        # otherwise find the middle index
        m = (left + right) >> 1

        # include the middel item if it's in range
        x = coords[2 * m]
        y = coords[2 * m + 1]
        if min_x <= x <= max_x and min_y <= y <= max_y:
            result.append(ids[m])

        # queue search in halves that intersect the query
        if min_x <= x if axis == 0 else min_y <= y:
            stack.append(left)
            stack.append(m - 1)
            stack.append(1 - axis)

        if max_x >= x if axis == 0 else max_y >= y:
            stack.append(m + 1)
            stack.append(right)
            stack.append(1 - axis)

    return result


# range ========================================================================


# within =======================================================================

# based on:
# https://github.com/mourner/kdbush/blob/ea3a81d272e1a87df3efe8c404021435dfa6cbfd/src/within.js#L2
def _within(
    ids: np.ndarray, coords: np.ndarray, qx: int, qy: int, r: int, node_size: int,
) -> List[int]:
    stack = [0, len(ids) - 1, 0]
    result = []
    r2 = r * r

    # recusively search for items within the radius in the kd-sorted arrays
    while len(stack):
        axis = stack.pop()
        right = stack.pop()
        left = stack.pop()

        # if we reach "tree node", search linearly
        if right - left <= node_size:
            for i in range(left, right + 1):
                if __sq_dist(coords[2 * i], coords[2 * i + 1], qx, qy) < r2:
                    result.append(ids[i])
            continue

        # otherwise find the middle index
        m = (left + right) >> 1

        x = coords[2 * m]
        y = coords[2 * m + 1]
        if __sq_dist(x, y, qx, qy) <= r2:
            result.append(ids[m])

        if qx - r <= x if axis == 0 else qy - r <= y:
            stack.append(left)
            stack.append(m - 1)
            stack.append(1 - axis)

        if qx + r >= x if axis == 0 else qy + r >= y:
            stack.append(m + 1)
            stack.append(right)
            stack.append(1 - axis)

    return result


def __sq_dist(ax: float, ay: float, bx: float, by: float) -> float:
    return (ax - bx) ** 2 + (ay - by) ** 2


# within =======================================================================
