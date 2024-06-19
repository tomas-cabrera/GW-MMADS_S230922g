import os
import os.path as pa
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

from scripts.utils import paths, plotting
from scripts.utils.light_curves import plot_light_curve, plot_light_curve_from_file
from scripts.utils.stamps import plot_stamp

###############################################################################

###############################################################################

# Load candidates_table
candidates_table_path = paths.tables / "candidates_table.tex"
if pa.exists(candidates_table_path):
    candidates_table = pd.read_csv(
        candidates_table_path,
        delimiter=" & ",
        names=[
            "objid",
            "z",
            "z_err",
            "z_source",
            "CR_2D",
            "CR_3D",
            "parsnip_class",
            "parsnip_prob",
        ],
        na_values=["-", "*"],
    )
    candidates_table["parsnip_prob"] = candidates_table["parsnip_prob"].apply(
        lambda x: x.split()[0]
    )
    candidates_table["z"] = pd.to_numeric(candidates_table["z"])

# Iterate over DECam photometry files
DECAM_DIR = paths.PHOTOMETRY_DIR / "DECam" / "diffphot"
figpaths = []
for oi, obj in enumerate(candidates_table["objid"]):
    # Get the first file that matches the object name
    glob_str = str(DECAM_DIR / f"*{obj}.fits")
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
            rasterized=True,
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
        rasterized=True,
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
        "rasterized": True,
    }
    kw_annotate = {
        "color": "xkcd:neon green",
        "fontsize": 8,
        "rasterized": True,
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
        plot_light_curve_from_file(fitsname, ax=ax, rasterized=True)
    else:
        plot_light_curve_from_file(fitsname, ax=ax, plot_legend=False, rasterized=True)

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
        rasterized=True,
    )

    # Set title
    axd["LC"].set_title(obj)
    plt.tight_layout()

    # Savefig and close
    figpath = paths.script_to_fig(
        __file__, key=f"{pa.basename(fitsname).replace('.fits', '')}"
    )
    os.makedirs(pa.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)
    plt.close()

    # Append to list of figpaths
    figpaths.append(figpath)

# Generate tex for figures

# Setup tex
figsetnum = 1
figpagestart_first = f"""
\\begin{{figure*}}
    \\centering
"""
figpageend_first = f"""
    \\caption{{Light curves for our remaining 40 candidates.}}
    \\label{{fig:light_curves_other}}
\\end{{figure*}}
"""
figpagestart = f"""
\\begin{{figure*}}\ContinuedFloat
    \\centering
"""
figpageend = f"""
    \\caption{{Light curves for our remaining 40 candidates (cont.).}}
    \\label{{fig:light_curves_other}}
\\end{{figure*}}
"""
n_fig_per_row = 2
n_rows_per_page = 3
figwidths_per_n = {1: 0.95, 2: 0.49, 3: 0.32, 4: 0.24}

# Iterate over figpaths
figsettex = ""
n_fig = 0
n_page = 0
n_fig_in_row = 0
n_rows_in_page = 0
for figpath in figpaths:
    # Get object name
    n_obj_pdf = pa.basename(figpath)
    obj = n_obj_pdf.split("_")[-1].replace(".pdf", "")

    ### Add string
    # Add start of page if appropriate
    if n_rows_in_page == 0 and n_fig_in_row == 0:
        if n_fig == 0:
            figsettex = figpagestart_first
        else:
            figsettex += figpagestart
    # Add start of row if appropriate
    if n_fig_in_row == 0:
        figsettex += f"""
    \\gridline{{"""
    # Add figure
    figsettex += f"""
    \\fig{{figures/light_curves_other/{n_obj_pdf}}}{{{figwidths_per_n[n_fig_per_row]}\\textwidth}}{{}}"""
    n_fig_in_row += 1
    # Add end of row if appropriate
    if n_fig_in_row == n_fig_per_row or n_fig == len(figpaths) - 1:
        figsettex += f"""
    }}"""
        n_fig_in_row = 0
        n_rows_in_page += 1
    # Add end of page if appropriate
    if n_rows_in_page == n_rows_per_page:
        if n_page == 0:
            figsettex += figpageend_first
        else:
            figsettex += figpageend
        n_page += 1
        n_rows_in_page = 0

    # Increment n_fig
    n_fig += 1

# Write to file
texpath = paths.script_to_fig(__file__, key="gridline").replace(".pdf", ".tex")
with open(texpath, "w") as f:
    f.write(figsettex)

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
texpath = paths.script_to_fig(__file__, key="figset").replace(".pdf", ".tex")
with open(texpath, "w") as f:
    f.write(figsettex)