import os.path as pa
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import sncosmo
from astropy.io import fits
from astropy.time import Time
from utils import paths, plotting
from utils.light_curves import plot_light_curve, plot_light_curve_from_file
from utils.stamps import plot_stamp

###############################################################################

###############################################################################

words = True

# Iterate over DECam photometry files
DECAM_DIR = paths.PHOTOMETRY_DIR / "DECam"
for oi, obj in enumerate(plotting.spectra_objs):
    # Get the first file that matches the object name
    glob_str = str(DECAM_DIR / f"*{obj}.fits")
    fitsname = glob(glob_str)[0]
    print(fitsname)

    # Initialize subplot mosaic
    fig, axd = plt.subplot_mosaic(
        [
            ["TEMP", "LC"],
            ["TEMP", "LC"],
            ["SCI", "LC"],
            ["SCI", "SPEC"],
            ["DIFF", "SPEC"],
            ["DIFF", "SPEC"],
        ],
        width_ratios=[1, 2],
        figsize=(7, 4),
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
        color="xkcd:neon green",
        ha="center",
        va="bottom",
    )
    axd["DIFF"].annotate(
        "E",
        xy=(x - bar - label_offset, y),
        color="xkcd:neon green",
        ha="right",
        va="center",
    )

    ### Light curve plot
    ax = axd["LC"]

    # Plot photometry
    plot_light_curve_from_file(fitsname, ax=ax, plot_legend=False)

    if not words:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", which="both", labelbottom=False, labeltop=False)
        ax.tick_params(axis="y", which="both", labelleft=False, labelright=False)

    # Move axis to top
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

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

    ax.relim()
    ax.autoscale()

    ### Spectra plot
    ax = axd["SPEC"]

    # Get spectra files; sort by date
    spectra_files = glob(str(paths.SPECTRA_DIR / f"*{obj}*.ascii"))
    spectra_files = sorted(spectra_files)

    # Iterate over spectra files
    count = 0
    for sf in spectra_files:
        # Load spectra
        spec = np.loadtxt(sf)
        wl = spec[:, 0]
        flux = spec[:, 1]
        flux_norm = flux / np.nanmedian(flux)
        flux_norm -= count
        yyyymmdd = pa.basename(sf).split("_")[1].split(".")[0]
        spectime = Time(yyyymmdd, format="isot").mjd

        # Make label
        label = f"{yyyymmdd}, {plotting.spectra_instruments[obj][yyyymmdd]}"

        # Plot spectra
        spectra_artist = ax.plot(
            wl,
            flux_norm,
            label=label,
            lw=1,
        )

        # Add vertical line to photometry plot
        axd["LC"].axvline(
            x=spectime,
            zorder=0,
            color=spectra_artist[0].get_color(),
        )

        count += 1

        # Add photometry for 6400 Keck spectrum
        if "C202309242206400m275139_2023-12-07" in sf:
            # Calculate photometry for spectrum
            bin_edges = np.zeros(len(wl) + 1)
            bin_edges[1:-1] = (wl[1:] + wl[:-1]) / 2
            bin_edges[0] = wl[0] - (wl[1] - wl[0]) / 2
            bin_edges[-1] = wl[-1] + (wl[-1] - wl[-2]) / 2
            spec_sncosmo = sncosmo.Spectrum(bin_edges=bin_edges, flux=flux)
            specphot = {"time": spectime}
            for b in np.unique(data["FILTER"]):
                # Ignore nan
                if b == "N/A":
                    continue

                try:
                    specphot[b] = spec_sncosmo.bandmag(f"des{b}", "ab")
                except ValueError as e:
                    print(f"Error: {e}")
                    continue

            # # Plot spectra photometry
            # for b, mag in specphot.items():
            #     continue
            #     if b == "time":
            #         continue
            #     print(b, mag)
            #     plot_light_curve(
            #         [
            #             specphot["time"],
            #         ],
            #         [
            #             specphot[b],
            #         ],
            #         [
            #             0,
            #         ],
            #         b,
            #         ax=axd["LC"],
            #         marker="x",
            #         markersize=8,
            #     )

    # Auxiliary
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    if words:
        ax.set_xlabel("Wavelength (A)")
        if len(spectra_files) > 1:
            ax.set_ylabel("Norm. flux + offset")
        else:
            ax.set_ylabel("Norm. flux")
        ax.legend(loc="lower right", frameon=False)

    if not words:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", which="both", labelbottom=False, labeltop=False)
        ax.tick_params(axis="y", which="both", labelleft=False, labelright=False)

    # Set title
    # fig.suptitle(obj)

    # Save figure
    plt.tight_layout()
    plt.savefig(
        __file__.replace("scripts", "tex/figures").replace(
            ".py", f"_{pa.basename(fitsname).replace('.fits', '')}" + ".png"
        )
    )
    plt.close()
