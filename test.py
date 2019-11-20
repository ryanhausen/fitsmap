import numpy as np
from astropy.io import fits

from fitsmap import mapmaker

if __name__ == "__main__":
    mapmaker.dir_to_map(
        "./tmp",
        out_dir="./tmp/web",
        zoom=4,
        exclude_predicate=lambda f: ".fits" in f,
        task_procs=2,
        procs_per_task=0,
        cat_wcs_fits_file="./tmp/test_mosaic_F277W_2019_06_18.fits",
    )
