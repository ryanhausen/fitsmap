.. Variables to ensure the hyperlink gets used
.. |convert| replace:: `fitsmap.convert <https://fitsmap.readthedocs.io/en/latest/source/fitsmap.html#module-fitsmap.convert>`__
.. |files_to_map| replace:: `fitsmap.convert.files_to_map <https://fitsmap.readthedocs.io/en/latest/source/fitsmap.html#fitsmap.convert.files_to_map>`__
.. |dir_to_map| replace:: `fitsmap.convert.dir_to_map <https://fitsmap.readthedocs.io/en/latest/source/fitsmap.html#fitsmap.convert.dir_to_map>`__

.. image:: docs/logo.svg.png
    :alt: FitsMap
    :align: center

FitsMap
=======

FitsMap is a tool for displaying astronomical images and their associated
catalogs, powered by `LeafletJS <https://leafletjs.com>`_.

Survey images can have dimensions in the tens of thousands of pixels in multiple
bands. Examining images of this size can be difficult, especially in multiple
bands. Memory constraints and highly specialized tools like DS9 can make simple
high-level analysis infeasible or cumbersome. FitsMap addresses these two
issues by converting large fits files and images into tiles that can be
presented using `LeafletJS <https://leafletjs.com>`_. Another issue in
examining survey images is examining a catalog of sources in the context of the
images. FitsMap addresses this by converting a catalog of sources into JSON map
markers, which can be viewed in the webpage. Additionally, these sources are
searchable using the web interface by the column ``id``.

Installation
************

Requirements:

- ``astropy``
- ``imageio``
- ``numpy``
- ``matplotlib``
- ``pillow``
- ``scikit-image``
- ``sharedmem``
- ``tqdm``

Use ``pip`` to install

.. code-block:: bash

    pip install fitsmap

Usage
*****

FitsMap is designed to address the following example. A user has multiple FITS
files or PNG files that represent multiple bands of the same area of the sky,
along with a catalog of sources within that area. For example, the directory
might look like:

::

  - path/to/data/
    - F125W.png
    - F160W.png
    - catalog.cat

To convert this diretory into a map is as simple as using |dir_to_map|:

.. code-block:: python

    from fitsmap import convert

    convert.dir_to_map.(
        "path/to/data",
        out_dir="/path/to/data/map",
        cat_wcs_fits_file="path/to/header_file.fits",
    )

The first argument is which directory contains the files that we would like to
convert into a map. In our case, this is ``path/to/dir``.  The next argument is
the ``out_dir`` keyword argument that tells FitsMap where to put the generated
webpage and supporting directories. In this example, the website will be built
in a new subdirectory called ``map`` within ``path/to/data``. Finally, the
last argument is the ``cat_wcs_fits_file`` keyword argument. This tells FitsMap
which header to use to parse any catalog files and convert them into map
markers. In this example, one of the FITS files in the directory is used.

Once FitsMap is finished, the following will have been generated:

::

  - path/to/data/map/
    - css/
    - F125W/
    - F160W/
    - js/
    - index.html

The directories ``F125W`` and ``F160W`` contain tiled versions of the input
fits files. The ``css`` directory contains some supporting css files for
clustering the markers. The ``js`` directory contains the json converted
catalog sources. Finally, ``index.html`` is the webpage that contains the map.
To use the map, simply open ``index.html`` with your favorite browser.

Parallelization *(Linux/Mac Only)*
**********************************

FitsMap supports the parallelization(via Multiprocessing/``sharedmem``) of map
creation in two ways:

- splitting images/catalogs into parallel tasks
- parallel tiling of an image

The settings for parallelization are set using the following keyword arguments:

- ``procs_per_task``: Sets how many layers/catalogs to convert in parallel at a
  time.
- ``task_procs``: How many tiles to generate in parallel

You can use both keyword arguments at the same time, but keep in mind the
number of cpus available. For example, if ``procs_per_task=2`` amd
``task_procs=2`` then that will generate 6 new processes, 2 new processes for
each task, and each of those will generate 2 new processes to tile an image in
parallel.

Parallelization offers a significant speed up, so if there are cores available
it makes sense to use them.

Notes
*****

Notes on Image Conversion
+++++++++++++++++++++++++

FitsMap has two "image engines" that you can choose from for converting
arrays into PNGS: PIL and Matplotlib.imshow. The default is to use PIL(pillow),
which seems to be faster but expects all files to be already normalized and
image ready. If the images are already normalized or are already PNGS, then
this will work fine. Matplotlib, although a little slower, can accept FITS
files without normalizing them. However, the default scaling is Linear and
changing it isn't currently supported. So images should  have their dynamic
range compressed before using FitsMap. Additionally, the default colomap passed
to imshow is "gray", but you can change this by changing the variable
``convert.MPL_CMAP`` to the string name of a
`Matplotlib colormap <https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html>`_.

To ensure that pixels are rendered correctly and that map markers are
placed correctly, any image that is not square is squared by padding the array
with NaN values that are converted into transparent pixels in the PNG. As a
consequence, if a FITS file contains NaNs when it is converted, those pixels
will be converted into transparent pixels.

Notes on Catalog Conversion
+++++++++++++++++++++++++++

Catalogs should be whitespace delimited text files with the first line
containing the column names, and the following lines containing values.
Catalogs need to have an ``id`` column with a unique value for each row. It
also needs to have coordinates for each source, which can be one of the
following pairs of columns (``ra``/``dec``) or (``x``/``y``).

All of the columns/values for each source will be stored in the description for
object and will show up when the marker is clicked. As a consequence,
having many columns will cause the following:

- Very large pop-up descriptions when a marker is clicked.
- Slower web page loading times due to the json marker file being larger.

----

For more information see the `docs <https://fitsmap.readthedocs.io>`__
or the `code <https://github.com/ryanhausen/fitsmap>`__.

