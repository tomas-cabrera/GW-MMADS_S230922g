import glob
import os
import os.path as pa

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table

###############################################################################


def plot_decam_wendelstein(wendelstein_path):
    ### Get DECam data
    objname = pa.basename(pa.dirname(wendelstein_path))
    globstr = pa.dirname(pa.dirname(pa.dirname(pa.dirname(wendelstein_path))))
    globstr += f"/decam/2024-01-03_shortlist/Candidates/*{objname}.fits"
    try:
        decam_path = glob.glob(globstr)[0]
    except IndexError:
        print(f"DECam data not found for {objname}")
        return

    with fits.open(decam_path) as hdul:
        decam_data = Table(hdul[1].data)

    ### Get Wendelstein data
    wendelstein_data = Table.read(wendelstein_path)
    wendelstein_data = wendelstein_data[wendelstein_data["MAG_APER_DIFF"] != 99]
    print(wendelstein_data["MJD_OBS"])
    print(wendelstein_data["MAG_APER_DIFF"])

    f2color = {
        "g": "green",
        "r": "red",
        "i": "orange",
        "z": "purple",
    }
    for f in set(
        list(set(wendelstein_data["FILTER"])) + list(set(decam_data["FILTER"]))
    ):
        if f not in f2color.keys():
            continue
        mask = decam_data["FILTER"] == f
        plt.scatter(
            decam_data[mask]["MJD_OBS"],
            decam_data[mask]["MAG_FPHOT"],
            color=f2color[f],
            marker="o",
        )
        mask = wendelstein_data["FILTER"] == f
        plt.scatter(
            np.array(wendelstein_data[mask]["MJD_OBS"].value),
            np.array(wendelstein_data[mask]["MAG_APER_DIFF"].value),
            color=f2color[f],
            marker="x",
        )
    plt.title(objname)
    plt.gca().invert_yaxis()
    plt.show()
    plt.close()


###############################################################################

wendelstein_dir = "/home/tomas/academia/projects/decam_followup_O4/S230922g/data/photometry/wendelstein/S230922g"
for wendelstein_path in glob.glob(f"{wendelstein_dir}/*/*.ecsv"):
    plot_decam_wendelstein(wendelstein_path)
