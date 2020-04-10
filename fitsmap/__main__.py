# MIT License
# Copyright 2020 Ryan Hausen

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
"""FitsMap CLI interface"""

import click

import fitsmap.convert as convert

HELP_OUT_DIR = "Output directory for map"
HELP_MIN_ZOOM = "Min zoom level for map"
HELP_TITLE = "HTML page title"
HELP_TASK_PROCS = "The number of tasks to run in parallel"
HELP_PROCS_PER_TASK = "The number of tiles to process in parallel"
HELP_CATALOG_DELIM = "The delimiter for .cat files, default is whitespace"
HELP_CAT_WCS_FITS_FILE = "FITS file with WCS for translating catalog ra/dec"
HELP_IMAGE_ENGINE = "Method to process image tiles PIL for pillow MPL for matplotlib"


@click.group()
def cli():
    """FitsMap --- Convert FITS files and catalogs into LeafletJS maps."""
    pass


@cli.command()
@click.argument("directory", type=str)
@click.option("--out_dir", default=".", help=HELP_OUT_DIR)
@click.option("--min_zoom", default=0, help=HELP_MIN_ZOOM)
@click.option("--title", default="FitsMap", help=HELP_TITLE)
@click.option("--task_procs", default=0, help=HELP_TASK_PROCS)
@click.option("--procs_per_task", default=0, help=HELP_PROCS_PER_TASK)
@click.option("--catalog_delim", default=None, help=HELP_CATALOG_DELIM)
@click.option("--cat_wcs_fits_file", default=None, help=HELP_CAT_WCS_FITS_FILE)
@click.option("--image_engine", default="PIL", help=HELP_IMAGE_ENGINE)
def dir(
    directory,
    out_dir,
    min_zoom,
    title,
    task_procs,
    procs_per_task,
    catalog_delim,
    cat_wcs_fits_file,
    image_engine,
):
    """--Convert a directory to a map.

    CLI interface: dir command.\n
    DIRECTORY the relative path to the directory to convert i.e. ./files/
    """
    convert.dir_to_map(
        directory,
        out_dir=out_dir,
        min_zoom=min_zoom,
        title=title,
        task_procs=task_procs,
        procs_per_task=procs_per_task,
        catalog_delim=catalog_delim,
        cat_wcs_fits_file=cat_wcs_fits_file,
        image_engine=image_engine,
    )


@cli.command()
@click.argument("files", type=str)
@click.option("--out_dir", default=".", help=HELP_OUT_DIR)
@click.option("--min_zoom", default=0, help=HELP_MIN_ZOOM)
@click.option("--title", default="FitsMap", help=HELP_TITLE)
@click.option("--task_procs", default=0, help=HELP_TASK_PROCS)
@click.option("--procs_per_task", default=0, help=HELP_PROCS_PER_TASK)
@click.option("--catalog_delim", default=None, help=HELP_CATALOG_DELIM)
@click.option("--cat_wcs_fits_file", default=None, help=HELP_CAT_WCS_FITS_FILE)
@click.option("--image_engine", default="PIL", help=HELP_IMAGE_ENGINE)
def files(
    files,
    out_dir,
    min_zoom,
    title,
    task_procs,
    procs_per_task,
    catalog_delim,
    cat_wcs_fits_file,
    image_engine,
):
    """--Convert a files to a map.

    CLI interface: files command.\n
    FILES should be a comma seperated list of files i.e. a.fits,b.fits,c.cat
    """
    convert.files_to_map(
        files.split(","),
        out_dir=out_dir,
        min_zoom=min_zoom,
        title=title,
        task_procs=task_procs,
        procs_per_task=procs_per_task,
        catalog_delim=catalog_delim,
        cat_wcs_fits_file=cat_wcs_fits_file,
        image_engine=image_engine,
    )
