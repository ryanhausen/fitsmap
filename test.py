import numpy as np
from astropy.io import fits

from fitsmap import mapmaker

if __name__ == "__main__":
    mapmaker.dir_to_map(
        "./tmp",
        out_dir="./tmp/web",
        zoom=None,
        exclude_predicate=lambda f: f.endswith(".fits"),
        task_procs=2,
        procs_per_task=2,
        cat_wcs_fits_file="./tmp/test_mosaic_F200W_2019_05_28.fits",
    )
