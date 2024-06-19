import os
import os.path as pa
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import sncosmo
import tol_colors as tc
from astropy.io import fits
from astropy.time import Time

import scripts.utils.light_curves as mylc
from scripts.utils import paths, plotting
from scripts.utils.stamps import plot_stamp

###############################################################################

INTERNAL_ZS = {
    "C202309242206400m275139": (0.184, 3.627e-5),
    "C202309242248405m134956": (0.128, 3.757e-5),
    "C202310042207549m253435": (0.248, 0.001),
}

###############################################################################

# Iterate over DECam photometry files
DECAM_DIR = paths.PHOTOMETRY_DIR / "DECam" / "diffphot"
for oi, obj in enumerate(plotting.spectra_objs):
    # Get the first file that matches the object name
    glob_str = str(DECAM_DIR / f"*{obj}.fits")
    fitsname = glob(glob_str)[0]
    print(fitsname)

    # Initialize subplot mosaic
    fig, axd = plt.subplot_mosaic(
        [
            ["LC", "TEMP"],
            ["LC", "TEMP"],
            ["LC", "SCI"],
            ["ALLPHOT", "SCI"],
            ["ALLPHOT", "DIFF"],
            ["ALLPHOT", "DIFF"],
            ["SPEC", "SPEC"],
            ["SPEC", "SPEC"],
            ["SPEC", "SPEC"],
        ],
        width_ratios=[3, 1],
        figsize=(7, 7.5),
        gridspec_kw={
            # "hspace": 0,
            "wspace": 0,
        },
    )

    ##############################
    ###     Stamps plots       ###
    ##############################

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

    ##############################
    ###    Light curve plot    ###
    ##############################

    ax = axd["LC"]

    # Plot photometry
    if oi == 0:
        mylc.plot_light_curve_from_file(fitsname, ax=ax)
    else:
        mylc.plot_light_curve_from_file(fitsname, ax=ax, plot_legend=False)

    # Move axis to top
    # ax.xaxis.set_label_position("top")
    # ax.xaxis.tick_top()

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

    # Plot vertical line indicating time of event
    ax.axvline(
        Time("2023-09-22T02:03:44", format="isot", scale="utc").mjd,
        color="k",
        linestyle="--",
        alpha=0.5,
        rasterized=True,
    )

    ##############################
    ###      Spectra plot      ###
    ##############################

    # Select axes
    ax = axd["SPEC"]

    # Get spectra files; sort by date
    spectra_files = glob(str(paths.SPECTRA_DIR / f"*{obj}*.ascii"))
    spectra_files = sorted(spectra_files)

    # Plot telluric lines
    telluric_lines = [
        [7594, 7621],
    ]
    color = tc.tol_cset("bright")[2]
    for tl in telluric_lines:
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
    text_height = 4
    text_height_offset = -0.1
    text_ha_offset = 50
    for name, params in spectral_lines.items():
        wl_shifted = [wl * (1 + INTERNAL_ZS[obj][0]) for wl in params["wl"]]
        for wls in wl_shifted:
            ax.axvline(wls, color="k", linestyle="--", alpha=0.3, rasterized=True)
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
            rasterized=True,
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
    ax.set_xlabel(r"Observed wavelength ($\rm \AA$)")
    if len(spectra_files) > 1:
        ax.set_ylabel("Norm. flux + offset")
    else:
        ax.set_ylabel("Norm. flux")
    ax.legend(loc="lower right", frameon=False)

    ##############################
    ###   All photometry plot  ###
    ##############################

    # Select axes
    ax = axd["ALLPHOT"]

    # Load light curves
    data_dict = {}
    for inst in plotting.kw_instruments.keys():
        try:
            df_temp = mylc.get_light_curve(obj, inst)
        except FileNotFoundError:
            continue
        data_dict[inst] = df_temp

    # Plot light curves
    artists_insts = {}
    artists_fs = {}
    for inst, df in data_dict.items():
        # Add instrument marker to artist list
        if inst not in artists_insts:
            (artists_insts[inst],) = ax.plot(
                [],
                [],
                label=inst,
                ls="",
                marker=plotting.kw_instruments[inst]["marker"],
                color="k",
            )

        # Iterate over filters
        for f in df["filter"].unique():
            # Skip 'N/A' rows
            if f == "N/A":
                continue

            # Add filter to artist list
            if f not in artists_fs:
                (artists_fs[f],) = ax.plot(
                    [],
                    [],
                    label=f,
                    ls="",
                    marker="o",
                    color=plotting.band2color[f],
                )

            # Mask to filter
            mask = df["filter"] == f

            # Get detection type
            # dettag = df["dettag"].iloc[0]

            # Get plot keywords
            # kw = plotting.kw_dettag[dettag]

            # Plot
            df_plot = df[mask]
            ax.errorbar(
                df_plot["mjd"],
                df_plot["mag"],
                yerr=df_plot["magerr"],
                ls="",
                lw=0.5,
                marker=plotting.kw_instruments[inst]["marker"],
                markersize=6,
                markeredgewidth=0,
                color=plotting.band2color[f],
                capsize=1.5,
                # **kw,
            )

    # Shade area of campaign plot
    ax.axvspan(
        *axd["LC"].get_xlim(),
        color="k",
        alpha=0.1,
    )

    ax.invert_yaxis()
    artists = {**artists_insts, **artists_fs}
    ax.set_xlabel("Time (MJD)")
    ax.set_ylabel("Magnitude")
    ax.legend(handles=artists.values(), frameon=True, labels=artists.keys())

    ##############################
    ###       Auxiliary        ###
    ##############################

    # Set title
    # fig.suptitle(obj)

    # Save figure
    plt.tight_layout()
    figpath = paths.script_to_fig(
        __file__, key=f"{pa.basename(fitsname).replace('.fits', '')}"
    )
    os.makedirs(pa.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)
    plt.close()
