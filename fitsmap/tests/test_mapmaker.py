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
"""Tests mapmaker.py"""

import os

import pytest

import fitsmap.mapmaker as mm
import fitsmap.tests.helpers as helpers


@pytest.mark.unit
def test_build_path():
    """Test the mapmaker.build_path function"""
    z, y, x = 1, 2, 3
    out_dir = helpers.TEST_PATH
    img_name = mm.build_path(z, y, x, out_dir)

    expected_img_name = os.path.join(out_dir, str(z), str(y), f"{x}.png")

    expected_file_name_matches = expected_img_name == img_name

    assert expected_file_name_matches


@pytest.mark.unit
def test_slice_idx_generator():
    """

    """
