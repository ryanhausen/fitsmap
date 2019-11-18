import numpy as np
from astropy.io import fits

from fitsmap import mapmaker

if __name__ == "__main__":
    mapmaker.dir_to_map(
        "./tmp",
        depth=5,
        multiprocessing_processes=2,
        out_dir="./tmp/web",
        cat_wcs_fits_file="./tmp/test_mosaic_F200W_2019_05_28.fits",
    )
