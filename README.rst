.. Variables to ensure the hyperlink gets used
.. |mapmaker| replace:: `fitsmap.mapmaker <https://fitsmap.readthedocs.io/en/latest/source/fitsmap.html#module-fitsmap.mapmaker>`__
.. |files_to_map| replace:: `fitsmap.mapmaker.files_to_map <https://fitsmap.readthedocs.io/en/latest/source/fitsmap.html#fitsmap.mapmaker.files_to_map>`__
.. |dir_to_map| replace:: `fitsmap.mapmaker.files_to_map <https://fitsmap.readthedocs.io/en/latest/source/itsmap.html#fitsmap.mapmaker.dir_to_map>`__

FitsMap
=======

FitsMap is a tool for displaying astronomical images and their associated
catalogs, powered by `LeafletJS <https://leafletjs.com>`_.

Installation
------------

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

Usuage
------

Import the |mapmaker| module

.. code-block:: python

    from fitsmap import mapmaker

Pass a list of files to |files_to_map|:

.. code-block:: python

    some_files = ...

    mapmaker.files_to_map(some_files)

OR, pass a directory to |dir_to_map|:

.. code-block:: python

    mapmaker.dir_to_map("path/to/files/")

For more informatio see the `docs <https://fitsmap.readthedocs.io>`__
or the `code <https://github.com/ryanhausen/fitsmap>`__.

