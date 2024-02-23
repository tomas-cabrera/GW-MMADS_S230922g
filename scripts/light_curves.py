import os.path as pa
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.lines import Line2D
from utils import paths, plotting

###############################################################################


def plot_light_curve(
    time,
    flux,
    flux_err,
    band,
    ax=None,
    band2color=plotting.band2color,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots()
    ax.errorbar(time, flux, yerr=flux_err, ls="", color=band2color[band], **kwargs)
    ax.set_xlabel("Time (MJD)")
    ax.set_ylabel("Mag")
    return ax


def plot_light_curve_from_file(
    fitsname,
    ax=None,
    band2color=plotting.band2color,
    plot_legend=True,
    **kwargs,
):
    with fits.open(fitsname) as hdul:
        data = hdul[1].data
        time = data["MJD_OBS"]
        flux = data["MAG_FPHOT"]
        flux_err = data["MAGERR_FPHOT"]

        if ax is None:
            fig, ax = plt.subplots()

        # Iterate over filters
        legend_elements = []
        for f in np.unique(data["FILTER"]):
            # Skip nan rows
            if f == "N/A":
                continue

            # Get filter mask
            filter_mask = data["FILTER"] == f
            mag_mask = data["MAG_FPHOT"] <= 25.0

            # Iterate over detection types
            for d in ["m", "q", "p"]:
                mask = filter_mask & mag_mask & (data["STATUS_FPHOT"] == d)
                plot_light_curve(
                    time[mask],
                    flux[mask],
                    flux_err[mask],
                    f,
                    ax=ax,
                    band2color=band2color,
                    **kwargs,
                    **plotting.kw_dettag[d],
                )

            # Add artists to legend
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=band2color[f],
                    label=f,
                    ls="",
                )
            )

        # Standardize ticks
        ax.set_xlim(60200, 60330)
        # x_spacing = 20
        # xlim = ax.get_xlim()
        # xlim = (np.floor(xlim[0] / x_spacing) * x_spacing, np.ceil(xlim[1] / x_spacing) * x_spacing)
        # ax.xaxis.set_ticks(np.arange(xlim[0], xlim[1], x_spacing)[1:])

        y_spacing = 0.5
        ylim = ax.get_ylim()
        ylim = (
            np.floor(ylim[0] / y_spacing) * y_spacing,
            np.ceil(ylim[1] / y_spacing) * y_spacing,
        )
        ax.yaxis.set_ticks(np.arange(ylim[0], ylim[1], y_spacing)[1:])

        ax.grid(True, c="xkcd:gray", alpha=0.5)
        ax.set_axisbelow(True)
        ax.invert_yaxis()

        # Add legend
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="k",
                marker="v",
                fillstyle="none",
                label="Upper limit",
                linestyle="",
            )
        )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="k",
                marker="s",
                fillstyle="none",
                alpha=0.5,
                label="Stamp",
                linestyle="",
                markersize=10,
            )
        )
        if plot_legend:
            ax.legend(handles=legend_elements, loc="lower center", ncols=2)
