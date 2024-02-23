import json
import os
import warnings

import healpy as hp
import ligo.skymap.plot
import matplotlib.pyplot as plt
import meander
import numpy as np
import pandas as pd
from astropy.table import QTable
from ligo.skymap import moc as lsm_moc
from skymap import Skymap, SurveySkymap
from skymap.constants import DECAM
from skymap.healpix import ang2disc
from utils import paths, plotting

###############################################################################

def compute_contours(proportions, samples):
    """Plot containment contour around desired level. E.g 90% containment of a
    PDF on a healpix map.

    Parameters:
    -----------
    proportions: list
        list of containment level to make contours for.
        E.g [0.68,0.9]
    samples: array
        array of values read in from healpix map
        E.g samples = hp.read_map(file)

    Returns:
    --------
    ra_list: list
        List of arrays containing RA values for desired contours [deg].
    dec_list: list
        List of arrays containing Dec values for desired contours [deg].
    """

    levels = []
    sorted_samples = list(reversed(list(sorted(samples))))
    # print(samples,sorted_samples)
    nside = hp.pixelfunc.get_nside(samples)
    sample_points = np.array(hp.pix2ang(nside, np.arange(len(samples)), nest=True)).T
    for proportion in proportions:
        level_index = (np.cumsum(sorted_samples) > proportion).tolist().index(True)
        level = (
            sorted_samples[level_index]
            + (sorted_samples[level_index + 1] if level_index + 1 < len(samples) else 0)
        ) / 2.0
        levels.append(level)
    contours_by_level = meander.spherical_contours(
        sample_points,
        samples,
        levels,
    )

    ra_list = []
    dec_list = []
    for contours in contours_by_level:
        for contour in contours:
            theta, phi = contour.T
            phi[phi < 0] += 2.0 * np.pi
            dec_list.append(90 - np.degrees(theta))
            ra_list.append(np.degrees(phi))
            # print(ra_list),print(dec_list)

    return ra_list, dec_list


def plot_coverage(
    obsplan_dir,
    gwmap,
    levels=[0.5, 0.9],
    angsize=3.0,
    grid_spacing=5,
    fil="g",
    fil_colors={
        "g": "xkcd:green",
        "r": "r",
        "i": "xkcd:goldenrod",
        "z": "m",
        "Y": "k",
    },
    level_linestyles=["-", "--", ":"],
):
    # Open skymap
    map_b = QTable.read(gwmap)

    # Diagnose skymap
    npix = len(map_b)
    nside = hp.npix2nside(npix)

    # Find coords of max prob
    maxpix = np.argmax(map_b["PROB"])
    ra_c, dec_c = hp.pix2ang(nside, np.arange(len(map_b)), lonlat=True, nest=True)
    ra_c = np.average(ra_c, weights=map_b["PROB"].value)
    dec_c = np.average(dec_c, weights=map_b["PROB"].value)

    # Initialize figure
    fig = plt.figure()
    ax = plt.gca()

    # Initialize skymap for coverage
    m = SurveySkymap(
        projection="cass",
        lon_0=ra_c,
        lat_0=dec_c,
        celestial=False,
        llcrnrlon=ra_c + angsize,
        llcrnrlat=dec_c - angsize,
        urcrnrlon=ra_c - angsize,
        urcrnrlat=dec_c + angsize,
    )

    # Draw grid
    grid_spacing = 5
    grid_presets = np.arange(0, 360, grid_spacing)
    grid_presets = np.concatenate([grid_presets, -grid_presets[1:]])
    meridians = [x for x in grid_presets if x > ra_c - angsize and x < ra_c + angsize]
    mers = m.draw_meridians(
        meridians,
        labelstyle="+/-",
        labels=[0, 0, 0, 1],
        rasterized=True,
    )
    # Convert meridians to [0,360]
    for k, mer in mers.items():
        mer[1][0].set_text(f"{k}°")
    parallels = [x for x in grid_presets if x > dec_c - angsize and x < dec_c + angsize]
    m.draw_parallels(
        parallels,
        labelstyle="+/-",
        labels=[1, 0, 0, 0],
        rasterized=True,
    )

    ### Plot skymap
    # Initialize pixels covering sky; for some reason np.arange doesn't work
    pixels = ang2disc(
        nside,
        ra_c,
        dec_c,
        2 * angsize,
    )
    # Get pixel probabilities from skymap
    values = hp.pixelfunc.reorder(map_b["PROB"].value, n2r=True)[pixels]
    # Plot
    m.draw_hpxmap(
        values,
        pixels,
        nside=nside,
        xsize=800,
        cmap="Purples",
        alpha=0.75,
        rasterized=True,
    )

    ### Plot contours
    nsidemax = 256
    if nside > nsidemax:
        map_b_coarse = hp.pixelfunc.ud_grade(
            map_b["PROB"].value, nsidemax, order_in="NESTED"
        )
        map_b_coarse = map_b_coarse / np.sum(map_b_coarse)
    else:
        map_b_coarse = map_b["PROB"].value
    ra_contour, dec_contour = compute_contours(levels, map_b_coarse)
    for level, ra, dec, ls in zip(
        levels, ra_contour, dec_contour, level_linestyles[: len(levels)]
    ):
        x, y = m.projtran(ra, dec)
        m.plot(
            x,
            y,
            color="xkcd:purple",
            lw=2,
            label=f"{level*100:.0f}%",
            ls=ls,
            alpha=0.5,
            rasterized=True,
        )

    ### Plot coverage
    for fn in os.listdir(obsplan_dir):
        if not fn.endswith(".csv"):
            continue
        df1 = pd.read_csv(os.path.join(obsplan_dir, fn))

        # Select filter
        df1 = df1[df1["fil"] == fil]

        # Plot fields
        # Current version of skymap uses a matrix when numpy thinks an array is appropriate
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.draw_focal_planes(
                df1["ra"],
                df1["dec"],
                color=fil_colors[fil],
                alpha=0.2,
                linewidths=20.0,
                edgecolors="k",
                rasterized=True,
            )

        # Draw circles around cherrypicked fields 
        if df1.shape[0] < 10:
            for _, row in df1.iterrows():
                m.tissot(
                    row["ra"],
                    row["dec"],
                    DECAM,
                    100,
                    facecolor="none",
                    edgecolor="k",
                    zorder=2,
                    alpha=0.5,
                    rasterized=True,
                )

    # Labels, legend
    ax.set_xlabel("RA [deg]", labelpad=20)
    ax.set_ylabel("dec [deg]", labelpad=30)
    ax.legend(title=f"{fil}-band", frameon=False, loc="upper right")

    # Save plot
    plt.tight_layout()
    plt.savefig(paths.script_to_fig(__file__, suffix=f"_{fil}"), dpi=300)


###############################################################################

obsplan_dir = f"{paths.data}/obsplan"

plot_coverage(obsplan_dir, paths.SKYMAP_FLATTENED, angsize=12, fil="g")
plot_coverage(obsplan_dir, paths.SKYMAP_FLATTENED, angsize=12, fil="i")
