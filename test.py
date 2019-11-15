import numpy as np
from astropy.io import fits

from fitsmap import mapmaker

if __name__ == "__main__":
    mapmaker.dir_to_map("./tmp", depth=3, multiprocessing_processes=2)
