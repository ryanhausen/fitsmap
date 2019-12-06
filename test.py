import numpy as np
from astropy.io import fits

from fitsmap import mapmaker

if __name__ == "__main__":

    mapmaker.dir_to_map(
        "./tmp",
        out_dir="./tmp/web",
        zoom=None,
        cat_wcs_fits_file="./tmp/F200Wfits.fits",
        # exclude_predicate=lambda f: f.endswith(".fits"),
        task_procs=4,
        procs_per_task=0,
        image_engine=mapmaker.IMG_ENGINE_MPL,
    )
