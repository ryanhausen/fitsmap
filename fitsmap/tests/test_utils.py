# MIT License
# Copyright 2021 Ryan Hausen and contributors

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
"""Tests utils.py"""
import os
import pytest

import fitsmap.utils as u
import fitsmap.tests.helpers as helpers


@pytest.mark.unit
@pytest.mark.cartographer
def test_build_digit_to_string():
    """test cartographer.build_digit_to_string"""
    digits = range(10)
    strings = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    ]

    for expected, actual in zip(strings, map(u.digit_to_string, digits)):
        assert expected == actual


@pytest.mark.unit
@pytest.mark.cartographer
def test_build_digit_to_string_fails():
    """test cartographer.build_digit_to_string"""
    digit = -1

    with pytest.raises(ValueError) as excinfo:
        u.digit_to_string(digit)

    assert "Only digits 0-9 are supported" in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.cartographer
def test_make_fname_js_safe_digit():
    """Test the cartographer.make_fname_js_safe functions."""

    unsafe = "123"
    expected = "one23"

    assert expected == u.make_fname_js_safe(unsafe)


@pytest.mark.unit
@pytest.mark.cartographer
def test_make_fname_js_safe_dot_dash():
    """Test the cartographer.make_fname_js_safe functions."""

    unsafe = "a.b-c"
    expected = "a_dot_b_c"

    assert expected == u.make_fname_js_safe(unsafe)


@pytest.mark.unit
@pytest.mark.cartographer
def test_make_fname_js_safe_no_change():
    """Test the cartographer.make_fname_js_safe functions."""

    safe = "abc"
    expected = "abc"

    assert expected == u.make_fname_js_safe(safe)


@pytest.mark.unit
@pytest.mark.cartographer
def test_make_fname_js_safe_no_change():
    """Test the cartographer.make_fname_js_safe functions."""

    expected_shape = (738, 480)

    helpers.setup(with_data=True)

    actual_shape = u.get_fits_image_size(
        os.path.join(helpers.TEST_PATH, "test_image.fits")
    )
    helpers.tear_down()

    assert expected_shape == actual_shape
