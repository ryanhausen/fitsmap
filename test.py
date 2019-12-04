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
        #image_engine=mapmaker.IMG_ENGINE_MPL
    )
