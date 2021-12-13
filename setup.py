"""
MIT License
Copyright 2020 Ryan Hausen

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


with open("./fitsmap/__version__.py", "r") as f:
    version = f.readlines()[0].strip().replace('"', "")

REQUIRES = [
    "astropy",
    "imageio",
    "numpy",
    "matplotlib",
    "pillow",
    "scikit-image",
    "sharedmem",
    "tqdm",
    "click",
    "mapbox_vector_tile",
]


setup(
    name="fitsmap",
    version=version,
    author="Ryan Hausen",
    author_email="rhausen@ucsc.edu",
    description=("Turn fits files/catalogs into a leafletjs map"),
    license="MIT",
    keywords="tools fits leaflet",
    url="https://github.com/ryanhausen/fitsmap",
    packages=find_packages(exclude="fitsmap.tests"),
    include_package_data=True,
    install_requires=REQUIRES,
    entry_points={"console_scripts": ["fitsmap=fitsmap.__main__:cli"]},
    long_description=read("README.rst"),
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: PyPy",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Development Status :: 4 - Beta",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering",
    ],
)
