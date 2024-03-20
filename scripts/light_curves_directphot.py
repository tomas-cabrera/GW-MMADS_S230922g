import os
import os.path as pa
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from utils import paths, plotting
from utils.light_curves import plot_light_curve, plot_light_curve_from_file
from utils.stamps import plot_stamp

###############################################################################

###############################################################################

# Iterate over DECam photometry files
# DECAM_DIR = paths.PHOTOMETRY_DIR / "DECam"
DECAM_DIR = "/home/tomas/academia/projects/decam_followup_O4/S230922g/data/decam/2024-01-03_shortlist/Candidates_directphot"
objs = os.listdir(DECAM_DIR)
objs = [o.split("_")[-1].split(".")[0] for o in objs]
figpaths = []
for oi, obj in enumerate(objs):
    # Get the first file that matches the object name
    glob_str = f"{DECAM_DIR}/*{obj}.fits"
    fitsname = glob(glob_str)[0]
    print(fitsname)

    # Initialize subplot mosaic
    fig, axd = plt.subplot_mosaic(
        [["LC", "TEMP"], ["LC", "SCI"], ["LC", "DIFF"]],
        width_ratios=[3, 1],
        figsize=(4, 2.5),
        gridspec_kw={
            "hspace": 0,
            "wspace": 0,
        },
    )

    ### Stamps plots
    # Open fits file
    with fits.open(fitsname) as hdul:
        data = hdul[1].data

        # Get max snr
        maxsnr_idx = np.nanargmax(data["SNR_FPHOT"])
        maxsnr_row = data[maxsnr_idx]

    # Plot stamps
    for stamp_str in ["TEMP", "SCI", "DIFF"]:
        # Select axes
        ax = axd[stamp_str]

        # Plot stamp
        plot_stamp(
            maxsnr_row[f"PixA_THUMB_{stamp_str}"],
            ax=ax,
            vmin=maxsnr_row[f"ZMIN_{stamp_str}"],
            vmax=maxsnr_row[f"ZMAX_{stamp_str}"],
        )

        # Add title
        ax.set_ylabel(stamp_str)
        ax.yaxis.set_label_position("right")

    ## Plot scale bar
    # Set coordinates
    bar = 10  # arcseconds
    arcsec_per_pix = 0.2637
    x = 30
    y = 110
    xmin = x - bar / arcsec_per_pix / 2
    xmax = x + bar / arcsec_per_pix / 2
    axd["DIFF"].plot([xmin, xmax], [y, y], color="xkcd:neon green", linewidth=1.5)
    axd["DIFF"].annotate(
        f'{bar}"',
        xy=(x, y - 5),
        # fontsize=12,
        color="xkcd:neon green",
        ha="center",
        va="bottom",
    )

    ## Plot compass
    x = 110
    y = 110
    bar = 20
    label_offset = 10
    kw_arrow = {
        "head_width": 5,
        "head_length": 7,
        "color": "xkcd:neon green",
    }
    kw_annotate = {
        "color": "xkcd:neon green",
        "fontsize": 8,
    }
    axd["DIFF"].arrow(
        x,
        y,
        0,
        -bar,
        **kw_arrow,
    )
    axd["DIFF"].arrow(
        x,
        y,
        -bar,
        0,
        **kw_arrow,
    )
    axd["DIFF"].annotate(
        "N",
        xy=(x, y - bar - label_offset),
        ha="center",
        va="bottom",
        **kw_annotate,
    )
    axd["DIFF"].annotate(
        "E",
        xy=(x - bar - label_offset, y),
        ha="right",
        va="center",
        **kw_annotate,
    )

    ### Light curve plot
    ax = axd["LC"]
    # Plot photometry
    if obj == "C202309242244079m290053":
        plot_light_curve_from_file(fitsname, ax=ax)
    else:
        plot_light_curve_from_file(fitsname, ax=ax, plot_legend=False)

    # Plot square indicating the time of the max snr
    ax.plot(
        maxsnr_row["MJD_OBS"],
        maxsnr_row["MAG_FPHOT"],
        color="k",
        marker="s",
        fillstyle="none",
        alpha=0.5,
        markersize=10,
        label="Stamp",
        linestyle="",
    )

    # Set title
    axd["LC"].set_title(obj)

    # Savefig and close
    figpath = paths.script_to_fig(
        __file__, suffix=f"_{pa.basename(fitsname).replace('.fits', '')}"
    )
    figpath = __file__.replace(
        "/GW-MMADS_S230922g/scripts",
        "/data/decam/2024-01-03_shortlist/light_curves_directphot",
    ).replace(".py", f"_{obj}.png")
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()

    # Append to list of figpaths
    figpaths.append(figpath)

# Generate figset tex for figures

# Setup tex
figsetnum = 1
figsetstart = f"""\\figsetstart
\\figsetnum{{{figsetnum}}}
\\figsettitle{{{pa.basename(__file__).replace(".py", "")}}}
"""
figsetend = f"""
\\figsetend"""

# Iterate over figpaths
figsettex = figsetstart
figsetgrpnum = 0
for figpath in figpaths:
    # Get object name
    obj = pa.basename(figpath).split("_")[-1].replace(".pdf", "")

    # Add string
    figsettex += f"""
\\figsetgrpstart
\\figsetgrpnum{{{figsetnum}.{figsetgrpnum}}}
\\figsetgrptitle{{{obj}}}
\\figsetplot{{{"/".join(figpath.split('/')[-2:])}}}
\\figsetgrpnote{{}}
\\figsetgrpend
"""

    # Increment figsetgrpnum
    figsetgrpnum += 1

# Add end
figsettex += figsetend

# Write to file
texpath = __file__.replace("/scripts/", "/tex/").replace(".py", ".figset.tex")
with open(texpath, "w") as f:
    f.write(figsettex)
