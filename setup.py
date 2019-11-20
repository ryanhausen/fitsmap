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

# https://pythonhosted.org/an_example_pypi_project/setuptools.html

import os
from setuptools import setup, find_packages


def read(fname):
    """Helper for README file."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


REQUIRES = [
    "astropy",
    "imageio",
    "numpy",
    "matplotlib",
    "pillow",
    "scikit-image",
    "sharedmem",
    "tqdm",
]


setup(
    name="fitsmap",
    version="0.0.3",
    author="Ryan Hausen",
    author_email="ryan.hausen@gmail.com",
    description=("Turn fits files/catalogs into a leafletjs map"),
    license="MIT",
    keywords="models tools",
    url="https://github.com/ryanhausen/fitsmap",
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIRES,
    long_description=read("README.rst"),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
    ],
)
