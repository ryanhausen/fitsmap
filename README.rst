.. Variables to ensure the hyperlink gets used
.. |convert| replace:: `fitsmap.convert <https://fitsmap.readthedocs.io/en/latest/source/fitsmap.html#module-fitsmap.convert>`__
.. |files_to_map| replace:: `fitsmap.convert.files_to_map <https://fitsmap.readthedocs.io/en/latest/source/fitsmap.html#fitsmap.convert.files_to_map>`__
.. |dir_to_map| replace:: `fitsmap.convert.dir_to_map <https://fitsmap.readthedocs.io/en/latest/source/fitsmap.html#fitsmap.convert.dir_to_map>`__

.. image:: https://raw.githubusercontent.com/ryanhausen/fitsmap/master/docs/logo.svg.png
    :alt: FitsMap
    :align: center

FitsMap
=======

.. image:: https://github.com/ryanhausen/fitsmap/actions/workflows/build-linux.yml/badge.svg
    :target: https://github.com/ryanhausen/fitsmap/actions/workflows/build-linux.yml

.. image:: https://github.com/ryanhausen/fitsmap/actions/workflows/build-osx.yml/badge.svg
    :target: https://github.com/ryanhausen/fitsmap/actions/workflows/build-osx.yml

.. image:: https://github.com/ryanhausen/fitsmap/actions/workflows/build-windows.yml/badge.svg
    :target: https://github.com/ryanhausen/fitsmap/actions/workflows/build-windows.yml

.. image:: https://readthedocs.org/projects/fitsmap/badge/?version=latest
    :target: https://fitsmap.readthedocs.io

.. image:: https://codecov.io/gh/ryanhausen/fitsmap/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/ryanhausen/fitsmap/

.. image:: https://img.shields.io/badge/python-3.7-blue.svg
    :target: https://www.python.org/downloads/release/python-370/

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black

.. image:: https://badgen.net/badge/doi/10.1016%2Fj.ascom.2022.100586/yellow
    :target: https://doi.org/10.1016/j.ascom.2022.100586


FitsMap is a tool developed in the `Computational Astrophysics Research Group
<https://robertson.sites.ucsc.edu/research/>`_ at UC Santa Cruz for displaying
astronomical images and their associated catalogs, powered by `LeafletJS
<https://leafletjs.com>`_.

Survey images can have dimensions in the tens of thousands of pixels in multiple
bands. Examining images of this size can be difficult, especially in multiple
bands. Memory constraints and highly specialized tools like DS9 can make simple
high-level analysis infeasible or cumbersome. FitsMap addresses these two issues
by converting large fits files and images into tiles that can be presented using
`LeafletJS <https://leafletjs.com>`_. Another issue in examining survey images
is examining a catalog of sources in the context of the images. FitsMap
addresses this by converting a catalog of sources into JSON map markers, which
can be viewed in the webpage. Additionally, these sources are searchable using
the web interface by the column ``id``.

Some sample websites that leverage FitsMap are:

- `DREaM Galaxy Catalogs <https://purl.org/fitsmap/dream>`_ `(Drakos, et al., 2022) <https://iopscience.iop.org/article/10.3847/1538-4357/ac46fb>`_
- `Morpheus <https://purl.org/fitsmap/morpheus>`_ `(Hausen & Robertson 2020) <https://iopscience.iop.org/article/10.3847/1538-4365/ab8868>`_

Additional examples are welcome! If you'd like to add your use case here, submit
an issue with the title "Use Case Example", and in the description, include the
URL to the FitsMap along with a title and also a link to an associated paper if
you'd like.


Here is an example using FitsMap to render a 32,727² image and ~33 million
sources from the `DReAM Galaxy Catalog <https://arxiv.org/abs/2110.10703>`_:

.. image:: https://raw.githubusercontent.com/ryanhausen/fitsmap/master/docs/dream_map.gif
    :alt: FitsMap
    :align: center

Installation
************

Requirements:

- ``astropy``
- ``cbor2``
- ``click``
- ``mapbox_vector_tile``
- ``matplotlib``
- ``numpy``
- ``pillow``
- ``scikit-image``
- ``ray``
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

.. code-block::

  - path/to/data/
    - F125W.fits
    - F160W.fits
    - RGB.png
    - catalog.cat

There are two FITS files (``F125W.fits``, ``F160W.fits``), a PNG file
(``RGB.png``), and catalog (``catalog.cat``) containing sources visible in the
image files. To render these files using FitsMap you can use
|dir_to_map| or |files_to_map|.

After the FitsMap has been generated you can view it in your web browser by
navigating to the directory containing the map (``index.html``) and running the
following command in the terminal:

.. code-block:: bash

    fitsmap serve

This will start up a webserver and open your browser to the page containing the
map. When your done with the map you can close your browser window and kill the
process running in the terminal.


``dir_to_map``
--------------

To convert this directory into a FitsMap using |dir_to_map|:

.. code-block:: python

    from fitsmap import convert

    convert.dir_to_map.(
        "path/to/data",
        out_dir="path/to/data/map",
        cat_wcs_fits_file="path/to/data/F125W.fits",
        norm_kwargs=dict(stretch="log", max_percent=99.5),
    )

The first argument is which directory contains the files that we would like to
convert into a map. In our case, this is ``path/to/dir``.  The next argument is
the ``out_dir`` keyword argument that tells FitsMap where to put the generated
webpage and supporting directories. In this example, the website will be built
in a new subdirectory called ``map`` within ``path/to/data``. The argument
``cat_wcs_fits_file`` keyword argument tells FitsMap which header to use to
parse any catalog files and convert them into map markers. The ``norm_kwargs``
argument should be a dictionary of kwargs that get passed to
`astropy.visulization.simple_norm
<https://docs.astropy.org/en/stable/api/astropy.visualization.mpl_normalize.simple_norm.html>`_
which is used to scale the FITS files before rendering.

Equivalently, using the FitsMap command line interface:

.. code-block::

  fitsmap dir --out_dir /path/to/data/map \
              --cat_wcs_fits_file "path/to/header_file.fits" \
              path/to/data

**Note:** The command line interface doesn't currently support ``norm_kwargs``.

Run ``fitsmap --help`` for more information


Once FitsMap is finished, the following will have been generated:

.. code-block::

  - path/to/data/map/
    - F125W/
    - F160W/
    - RGB/
    - catalog/
    - css/
    - catalog_assets/
    - imgs/
    - js/
    - index.html

The directories ``F125W``, ``F160W``, ``RGB``, ``catalog`` contain tiled
versions of the input fits files. The ``css`` directory contains some supporting
CSS files for clustering the markers and rendering pixels. The ``imgs``
directory contains supporting images. The ``js`` directory contains supporting
JavaScript for the map. ``catalog_assets`` contains JSON files for each source
in each that are rendered when the marker associated with that source is
clicked. Finally, ``index.html`` is the webpage that contains the map.

To use the map, run ``fitsmap serve`` in the same directory as ``index.html``


``files_to_map``
----------------

If you want to specify the files that get used to generate the map you can use
function |files_to_map|:

.. code-block:: python

    from fitsmap import convert

    paths_to_files = [
        ...,
    ]

    convert.files_to_map.(
        paths_to_files,
        out_dir="path/to/data/map",
        cat_wcs_fits_file="path/to/header_file.fits",
        norm_kwargs=dict(stretch="log", max_percent=99.5),
    )

This will produce a map in ``out_dir`` using the files that were passed in using
the ``paths_to_files`` variable.


File Specific ``norm_kwargs``
-----------------------------

The ``norm_kwargs`` argument to |dir_to_map| and |files_to_map| can be a
dictionary of kwargs where the keys are the filenames (not paths) and the values
are the ``simple_norm`` kwargs for that file. For example:

.. code-block:: python

    from fitsmap import convert

    paths_to_files = [
        "fits_images/F125W.fits",
        "fits_images/F160W.fits",
    ]

    convert.files_to_map.(
        paths_to_files,
        out_dir="path/to/data/map",
        cat_wcs_fits_file="path/to/header_file.fits",
        norm_kwargs={
            "F125W.fits":dict(stretch="log", max_percent=99.5),
            "F160W.fits":dict(stretch="log", max_percent=99.9, min_percent=0.1),
        }
    )


Saveable Views
**************

FitsMap stores the current view (location/zoom) in the url. You can then
share the view with others by sharing the url.


Search
**************

You can search the catalogs by the ``id`` column from the catalog and FitsMap
will locate and pan to the source in the map.


Parallelization
**********************************

FitsMap supports the parallelization(via `ray <ray.io>`_) of map creation in two
ways:

- splitting images/catalogs into parallel tasks
- parallel tiling of an image
- parallel reading/tiling of a catalog

The settings for parallelization are set using the following keyword arguments:

- ``procs_per_task``: Sets how many layers/catalogs to convert in parallel at a
  time.
- ``task_procs``: How many processes can work on a single task.

You can use both keyword arguments at the same time, but keep in mind the number
of CPUs available. For example, if ``procs_per_task=2`` and ``task_procs=2``
then that will generate 6 new processes, 2 new processes for each task, and each
of those will generate 2 new processes to tile an image in parallel.

Parallelization can offer a significant speed up, so if there are cores available
it makes sense to use them.

**NOTE: ray's support for Windows is currently in beta, so you may experience
some bugs running in parallel on Windows machines. Feel free to submit an issue
if you run into any problems.**

Notes
*****

Notes on Image Conversion
-------------------------

FITS images are rendered into PNG map tiles using Matplotlib colormaps. The
default colormap used when rendering tiles is "gray". This can be changed by
setting the value of ``convert.MPL_CMAP`` to any valid `Matplotlib colormap
<https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html>`_.

To ensure that pixels are rendered correctly and that map markers are placed
correctly, any image that is not square is squared by padding the array with NaN
values that are converted into transparent pixels in the PNG. As a consequence,
if a FITS file contains NaNs when it is converted, those pixels will be
converted into transparent pixels.

Notes on Catalog Conversion
---------------------------

Catalogs should be delimited text files with the first line containing the
column names, and the following lines containing values. Catalogs need to have
an ``id`` column with a unique value for each row. It also needs to have
coordinates for each source, which can be one of the following pairs of columns
(``ra`` / ``dec``) or (``x`` / ``y``). **Note fitsmap assumes that the origin of
the image starts at (1,1), but this can be changed to (0,0) by setting the
kwarg** ``catalog_starts_at_one=False``.

Some catalogs have many columns for each row, which will create very tall
pop-ups when the markers are clicked. To avoid this, you can pass an integer
value using ``rows_per_column`` to either |dir_to_map| or |files_to_map|. This
will break the information into ``rows_per_column`` sized columns.

Catalog pop-ups are rendered as a simple HTML table, so you can put any HTML
friendly things, for example <img> tags, in the catalog and they should be
rendered appropriately.

FitsMap will render your markers as Ellipses if you have the following columns
in your catalog: ``a``, ``b``, and ``theta``. Where ``a`` is the major axis
radius in **pixels**, ``b`` is the minor axis radius in **pixels**, and theta
is the rotation of the ellipse in units of degrees starting from the negative
x-axis and moving counter-clockwise.

.. image:: https://raw.githubusercontent.com/ryanhausen/fitsmap/master/docs/ellipse_fig.png
    :alt: EllipseOrientaton
    :align: center

----

If you use FitsMap in your research please cite it using the following (also in
`CITE.bib <https://github.com/ryanhausen/fitsmap/blob/master/CITE.bib>`_):

.. code-block::

    @article{hausen2022a,
         title = {FitsMap: A simple, lightweight tool for displaying interactive astronomical image and catalog data},
       journal = {Astronomy and Computing},
        volume = {39},
         pages = {100586},
          year = {2022},
          issn = {2213-1337},
           doi = {https://doi.org/10.1016/j.ascom.2022.100586},
           url = {https://www.sciencedirect.com/science/article/pii/S2213133722000257},
        author = {R. Hausen and B.E. Robertson},
      keywords = {Astronomy web services (1856), Astronomy data visualization (1968), Astronomy data analysis (1858), Human-centered computing Scientific visualization (10003120.10003145.10003147.10010364), Human-centered computing Visualization toolkits (10003120.10003145.10003151.10011771)},
      abstract = {The visual inspection of image and catalog data continues to be a valuable aspect of astronomical data analysis. As the scale of astronomical image and catalog data continues to grow, visualizing the data becomes increasingly difficult. In this work, we introduce FitsMap, a simple, lightweight tool for visualizing astronomical image and catalog data. FitsMap uses well-understood image tiling techniques and a novel catalog tiling technique to serve gigapixel images with catalogs containing tens of millions of sources using only a simple web server. Further, the web-based visualizations can be viewed performantly on mobile devices. FitsMap is implemented in Python and is open source (https://github.com/ryanhausen/fitsmap).}
    }



For more information see the `docs <https://fitsmap.readthedocs.io>`__
or the `code <https://github.com/ryanhausen/fitsmap>`__.


