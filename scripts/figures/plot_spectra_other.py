import os
import os.path as pa
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
import tol_colors as tc

from scripts.utils import paths, plotting
from scripts.utils.light_curves import plot_light_curve, plot_light_curve_from_file
from scripts.utils.stamps import plot_stamp

###############################################################################

INTERNAL_ZS = {
    "C202309242206400m275139": (0.184, 3.627e-5),
    "C202309242248405m134956": (0.128, 3.757e-5),
    "C202310042207549m253435": (0.248, 0.001),
    "A202310262246341m291842": (0.15, 0.001),
}

###############################################################################

# Load candidates_table
candidates_table_path = paths.tables / "candidates_table.tex"
if pa.exists(candidates_table_path):
    candidates_table = pd.read_csv(
        candidates_table_path,
        delimiter=" & ",
        names=[
            "tnsid",
            "z",
            "z_err",
            "z_source",
            "CR_2D",
            "CR_3D",
            "parsnip_class",
            "parsnip_prob",
        ],
        na_values=["-", "*"],
        skiprows=14,
        skipfooter=2,
        engine="python",
    )
    candidates_table["parsnip_prob"] = candidates_table["parsnip_prob"].apply(
        lambda x: x.split()[0]
    )
    candidates_table["z"] = pd.to_numeric(candidates_table["z"])

# Iterate over DECam photometry files
DECAM_DIR = paths.PHOTOMETRY_DIR / "DECam" / "diffphot"
figpaths = []
for oi, tns in enumerate(candidates_table["tnsid"]):
    # Get objid
    for objid, tnsid in plotting.tns_names.items():
        if tnsid == tns:
            obj = objid
            break

    ##############################
    ###      Spectra plot      ###
    ##############################

    # Get spectra files; sort by date
    spectra_files = glob(str(paths.SPECTRA_DIR / f"*{obj}*.ascii"))
    spectra_files = sorted(spectra_files)

    # Skip if no spectra found
    if len(spectra_files) == 0:
        continue

    # Initialize figure, axes
    fig = plt.figure(figsize=(7, 2.5))
    ax = fig.add_subplot()

    # Iterate over spectra files
    count = 0
    wl_min = np.inf
    wl_max = -np.inf
    for sf in spectra_files:
        # Load spectra
        spec = np.loadtxt(sf)
        wl = spec[:, 0]
        flux = spec[:, 1]

        # Clip data for C202309242248405m134956 DBSP
        if obj == "C202309242248405m134956":
            mask = (wl > 4000) & (wl < 10000)
            wl = wl[mask]
            flux = flux[mask]

        flux_norm = flux / np.nanmedian(flux)
        flux_norm -= count
        yyyymmdd = pa.basename(sf).split("_")[1].split(".")[0]
        spectime = Time(yyyymmdd, format="isot").mjd

        # Save min/max wavelength
        wl_min = min(wl_min, np.min(wl))
        wl_max = max(wl_max, np.max(wl))

        # Make label
        label = f"{yyyymmdd}, {plotting.spectra_instruments[obj][yyyymmdd]}"

        # Plot spectra
        spectra_artist = ax.plot(
            wl,
            flux_norm,
            label=label,
            lw=1,
        )

        count += 1

    # Plot telluric lines
    telluric_lines = [
        [6867, 6884],
        [7594, 7621],
    ]
    color = tc.tol_cset("bright")[2]
    for tl in telluric_lines:
        if tl[0] < wl_max or tl[1] > wl_min:
            ax.axvspan(tl[0], tl[1], color=color, alpha=0.2, rasterized=True)

    # Plot spectral lines
    spectral_lines = {
        "MgII": {
            "wl": [2799],
            "ha": "left",
        },
        r"H$\beta$": {
            "wl": [4861],
            "ha": "right",
        },
        "OIII": {
            "wl": [5007, 4959],
            "ha": "left",
        },
        r"H$\alpha$": {
            "wl": [6563],
            "ha": "right",
        },
        "NII": {
            "wl": [6584],
            "ha": "left",
        },
    }
    # Set text parameters
    text_ha_offset = 50
    text_height_frac = 0.8
    text_height_offset_frac = -0.01
    ylim = ax.get_ylim()
    text_height = ylim[0] + text_height_frac * (ylim[1] - ylim[0])
    text_height_offset = (ylim[1] - ylim[0]) * text_height_offset_frac
    for name, params in spectral_lines.items():
        # Shift by redshift
        wl_shifted = [wl * (1 + INTERNAL_ZS[obj][0]) for wl in params["wl"]]

        # Skip if not in wavelength range
        if not any([wls > wl_min and wls < wl_max for wls in wl_shifted]):
            continue

        # Skip for A202310262246341m291842
        if obj == "A202310262246341m291842":
            continue

        # Plot lines
        for wls in wl_shifted:
            ax.axvline(wls, color="k", linestyle="--", alpha=0.3, rasterized=True)

        # Annotate lines
        ax.text(
            wl_shifted[0] + text_ha_offset * {"left": 1, "right": -1}[params["ha"]],
            text_height + text_height_offset,
            name,
            ha=params["ha"],
            va="center",
            color="k",
            fontsize=8,
            # rotation=90,
            # backgroundcolor="w",
            # bbox=dict(facecolor="w", edgecolor="none", alpha=0.5),
            rasterized=True,
        )
        text_height_offset *= -1

    # Auxiliary
    ax.set_xlabel(r"Observed wavelength ($\rm \AA$)")
    if len(spectra_files) > 1:
        ax.set_ylabel("Norm. flux + offset")
    else:
        ax.set_ylabel("Norm. flux")
    ax.legend(loc="upper right", frameon=False)

    # Set title
    ax.set_title(tns)
    plt.tight_layout()

    # Savefig and close
    figpath = paths.script_to_fig(
        __file__,
        key=obj,
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
    \\caption{{
        Additional spectra taken as a part of our follow-up campaign.
        A selection of spectral lines and telluric features are marked as in Figure \ref{{fig:C202309242206400m275139}}.
    }}
    \\label{{fig:{pa.basename(__file__).replace(".py", "")}}}
\\end{{figure*}}
"""
figpagestart = f"""
\\begin{{figure*}}\ContinuedFloat
    \\centering
"""


def figpageend(page_no):
    return f"""
    \\caption{{
        Light curves for our remaining {len(figpaths) - 1} candidates (cont.).
    }}
    \\label{{fig:light_curves_other.{page_no}}}
\\end{{figure*}}
"""


n_fig_per_row = 1
n_rows_per_page = 3
figwidths_per_n = {1: 0.95, 2: 0.49, 3: 0.32, 4: 0.24}

# Iterate over figpaths
figsettex = ""
n_fig_added = 0
n_fig_skipped = 0
n_page = 0
n_fig_in_row = 0
n_rows_in_page = 0
for figpath in figpaths:
    # Get object name
    obj_pdf = pa.basename(figpath)
    obj = obj_pdf.replace(".pdf", "")

    # Skip if favored object
    if obj == "C202309242206400m275139":
        n_fig_skipped += 1
        continue

    ### Add string
    # Add start of page if appropriate
    if n_rows_in_page == 0 and n_fig_in_row == 0:
        if n_fig_added == 0:
            figsettex = figpagestart_first
        else:
            figsettex += figpagestart
    # Add start of row if appropriate
    if n_fig_in_row == 0:
        figsettex += f"""
    \\gridline{{"""
    # Add figure
    figsettex += f"""
    \\fig{{figures/plot_spectra_other/{obj_pdf}}}{{{figwidths_per_n[n_fig_per_row]}\\textwidth}}{{}}"""
    n_fig_in_row += 1
    # Add end of row if appropriate
    if (
        n_fig_in_row == n_fig_per_row
        or n_fig_added == len(figpaths) - n_fig_skipped - 1
    ):
        figsettex += f"""
    }}"""
        n_fig_in_row = 0
        n_rows_in_page += 1
    # Add end of page if appropriate
    if (
        n_rows_in_page == n_rows_per_page
        or n_fig_added == len(figpaths) - n_fig_skipped - 1
    ):
        if n_page == 0:
            figsettex += figpageend_first
        else:
            figsettex += figpageend(n_page)
        n_page += 1
        n_rows_in_page = 0

    # Increment n_fig
    n_fig_added += 1

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
