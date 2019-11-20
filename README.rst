FitsMap
=======

FitsMap is a tool for displaying astronomical images and their associated
catalogs, powered by `LeafletJS <https://leafletjs.com>`_.

Installation
------------

Requirements:

- `astropy`
- `imageio`
- `numpy`
- `matplotlib`
- `pillow`
- `scikit-image`
- `sharedmem`
- `tqdm`

Use ``pip`` to install

.. code-block:: bash

    pip install fitsmap

Usuage
------

Import the ``mapmaker`` module

.. code-block:: python

    from fitsmap import mapmaker


Pass a list of files to ``files_to_map``:

.. code-block:: python

    some_files = ...

    mapmaker.files_to_map(some_files)

OR, pass a directory to ``dir_to_map``:

.. code-block:: python

    mammaker.dir_to_map("path/to/files/")


