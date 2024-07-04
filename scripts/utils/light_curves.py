import datetime
import os.path as pa
from copy import copy
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sncosmo
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from matplotlib.lines import Line2D
from scipy.stats import norm

from scripts.utils import paths, plotting

###############################################################################


# Source: https://noirlab.edu/science/programs/ctio/filters/Dark-Energy-Camera
band2central_wavelength = {
    "u": 355.0 * u.nm,
    "g": 473.0 * u.nm,
    "r": 642.0 * u.nm,
    "i": 784.0 * u.nm,
    "z": 926.0 * u.nm,
    "Y": 1009.0 * u.nm,
    "VR": 626.0 * u.nm,
}

KIMURA20_SF_PARAMS = pd.read_csv(
    paths.data / "photometry" / "kimura20_SFparams.dat", sep="\s+"
).set_index("band")


def get_bandpass(filter):
    """Generic function to get bandpass data for a given filter.
    DECam source: https://www.ctio.noirlab.edu/noao/node/13140

    Parameters
    ----------
    filter : str
        A string of the format f"{instrument}-{filter}".
        Can also be "sncosmo-{key}" to get the respective data from the sncomsmo package.

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    # Parse filter
    instrument, fil = filter.split("-")

    # Load data
    if instrument == "DECam":
        bandpass = pd.read_csv(
            paths.data / "photometry" / "DECam" / "bandpasses.txt", sep="\s+"
        )
        bandpass = bandpass[["#LAMBDA", fil]]
        bandpass.rename(
            columns={"#LAMBDA": "lambda", fil: "transmission"}, inplace=True
        )
    elif instrument == "sncosmo":
        bandpass = copy(sncosmo.get_bandpass(fil))
        bandpass = pd.DataFrame(
            {"lambda": bandpass.wave / u.AA, "transmission": bandpass(bandpass.wave)}
        )
    else:
        raise ValueError(f"Unknown instrument: {instrument}")

    # Return
    return bandpass


def mag2fluxdensity(
    mag,
    band=None,
    mag_unit=u.ABmag,
    fluxdensity_unit=u.erg / u.s / u.cm**2 / u.AA,
    central_wavelength=None,  # Central wavelength of the band
):
    # Get central wavelength if needed
    if central_wavelength is None:
        central_wavelength = band2central_wavelength[band]

    mag_temp = mag * mag_unit
    return mag_temp.to(fluxdensity_unit, u.spectral_density(central_wavelength)).value


def fluxdensity2mag(
    fluxdensity,
    band=None,
    fluxdensity_unit=u.erg / u.s / u.cm**2 / u.AA,
    mag_unit=u.ABmag,
    central_wavelength=None,  # Central wavelength of the band
):
    # Get central wavelength if needed
    if central_wavelength is None:
        central_wavelength = band2central_wavelength[band]

    fluxdensity_temp = fluxdensity * fluxdensity_unit
    return fluxdensity_temp.to(mag_unit, u.spectral_density(central_wavelength)).value


def calc_sf_chance_probability_kimura20(
    deltat,
    deltamag,
    band,
):
    prob = calc_sf_chance_probability(
        deltat,
        deltamag,
        band,
        sf0=KIMURA20_SF_PARAMS.loc[band, "SF0"],
        t0=KIMURA20_SF_PARAMS.loc[band, "dt0"],
        bt=KIMURA20_SF_PARAMS.loc[band, "bt"],
    )

    return prob


def calc_sf_chance_probability(
    deltat,
    deltamag,
    band,
    sf0,
    t0,
    bt,
):
    # Calculate SF for deltat
    sf_deltat = sf0 * (deltat / t0) ** bt

    # Calculate probability of deltamag given SF
    prob = 1 - norm(0, sf_deltat).cdf(deltamag)

    return prob


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
    ax.set_ylabel("Magnitude")
    return ax


def plot_light_curve_from_file(
    fitsname,
    ax=None,
    band2color=plotting.band2color,
    plot_legend=True,
    **kwargs,
):
    with fits.open(fitsname) as hdul:
        # Load data
        data = hdul[1].data

        # Remove bad expnames
        mask_is_bad_expname = np.array(
            [s in plotting.BAD_EXPNAMES for s in data["SCIENCE_NAME"]]
        )
        data = data[~mask_is_bad_expname]

        # Get time, flux, flux_err
        time = data["MJD_OBS"]
        flux = data["MAG_FPHOT"]
        lim_mag_5 = data["LIM_MAG5"]
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

            # Iterate over detection types
            legend_include_upper_limit = False
            for d in ["m", "q", "p"]:
                # Initialize marker
                marker = None

                # Select data
                if d == "m":
                    y = lim_mag_5
                    y_err = np.zeros_like(y)
                    marker = "v"
                    markersize = 5
                else:
                    y = flux
                    y_err = flux_err
                # Define mask
                mask = filter_mask & (y < 25) & (data["STATUS_FPHOT"] == d)

                # Define marker
                if marker is None:
                    if f == "g":
                        marker = "X"
                        markersize = 5
                    elif f == "i":
                        marker = "P"
                        markersize = 5

                if np.any(mask):
                    if d == "m":
                        legend_include_upper_limit = True
                    # Plot
                    plot_light_curve(
                        time[mask],
                        y[mask],
                        y_err[mask],
                        f,
                        ax=ax,
                        band2color=band2color,
                        marker=marker,
                        **kwargs,
                        **plotting.kw_dettag[d],
                    )

            # Add artists to legend
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color=band2color[f],
                    label=f"DECam {f}",
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
        if legend_include_upper_limit:
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
            if len(legend_elements) > 3:
                ncols = 2
            else:
                ncols = 1
            ax.legend(
                handles=legend_elements,
                ncols=ncols,
                # loc="lower center",
                frameon=True,
            )


def get_light_curve_path(objid, instrument, wise_data="Processed_Data"):
    ### Get path
    lc_path = None
    instpath = f"{paths.data}/photometry/{instrument}"
    if instrument == "DECam":
        globstr = f"{instpath}/*_{objid}.fits"
        lc_path = paths.glob_plus(globstr, require_one=True)
    elif instrument == "SkyMapper":
        lc_path = f"{instpath}/{objid}.dat"
    elif instrument == "Wendelstein":
        lc_path = f"{instpath}/{paths.EVENTNAME}/{objid}/{objid}.ecsv"
    elif instrument == "WISE":
        # Try to get TNS name
        try:
            objid_wise = plotting.tns_names[objid]
        except KeyError:
            objid_wise = objid
        globstr = f"{instpath}/{objid_wise}*"
        wise_obj_path = paths.glob_plus(globstr, require_one=True)
        wise_objid = pa.basename(wise_obj_path)
        if wise_data == "Processed_Data":
            globstr = (
                f"{instpath}/{wise_objid}/{wise_objid}/{wise_data}/*_Mags_File.ascii"
            )
            lc_path = paths.glob_plus(globstr)
        elif wise_data == "Raw_Data":
            globstr = (
                f"{instpath}/{wise_objid}/{wise_objid}/Raw_Data/{wise_objid}*.ascii"
            )
            lc_path = paths.glob_plus(globstr)
    else:
        raise ValueError(f"Unknown instrument: {instrument}")

    return lc_path


def get_light_curve(objid, instrument):
    # Get path
    lc_path = get_light_curve_path(objid, instrument)

    # Load data
    if type(lc_path) is str:
        if instrument == "DECam":
            with fits.open(lc_path) as hdul:
                data = hdul[1].data
                mjd = data["MJD_OBS"]
                mag = data["MAG_FPHOT"]
                magerr = data["MAGERR_FPHOT"]
                filter = data["FILTER"]
        elif instrument == "SkyMapper":
            data = pd.read_csv(lc_path, sep="\s+")

            def yyyymmddhhmmss_to_isot(yyyymmddhhmmss):
                return datetime.datetime.strptime(
                    yyyymmddhhmmss, "%Y%m%d%H%M%S"
                ).isoformat()

            def yyyymmddhhmmss_to_mjd(yyyymmddhhmmss):
                return Time(yyyymmddhhmmss_to_isot(yyyymmddhhmmss), format="isot").mjd

            mjd = data["ImageID"].apply(lambda x: yyyymmddhhmmss_to_mjd(str(x)))
            mag = data["Magnitude_APC05"]
            magerr = data["MAgnitude_APC05_err"]
            filter = data["Filter"]
        elif instrument == "Wendelstein":
            data = Table.read(lc_path, format="ascii.ecsv")
            mjd = data["MJD_OBS"].value
            mag = data["MAG_AUTO_DIFF"].value
            magerr = data["MAGERR_AUTO_DIFF"].value
            filter = data["FILTER"].value
        else:
            raise ValueError(f"Unknown instrument: {instrument}")
    elif type(lc_path) is list:
        mjd = []
        mag = []
        magerr = []
        filter = []
        for p in lc_path:
            if instrument == "WISE":
                data = Table.read(p, format="ascii")
                if "Processed_Data" in p:
                    # Get band name
                    band = pa.basename(p).split("_")[2]
                    # Get data
                    mjd.extend(data[f"{band}MJD"])
                    mag.extend(data[f"{band}ApparentMag"])
                    magerr.extend(data[f"{band}ApparentMagErr"])
                    filter.extend([band] * len(data))
                elif "Raw_Data" in p:
                    for i in list("12"):
                        # Get column names
                        magcols = [f"w{i}mpro", f"w{i}pro_ep"]
                        j = 0
                        try:
                            while magcols[j] not in data.colnames:
                                j += 1
                        except IndexError:
                            raise ValueError(f"Could not find mag column for W{i}")
                        magcol = magcols[j]
                        magerrcol = magcol.replace("mpro", "sigmpro")
                        # Get data
                        mjd.extend(data["mjd"])
                        mjd.extend(data[f"mjd"])
                        mag.extend(data[magcol])
                        magerr.extend(data[magerrcol])
                        filter.extend([f"W{i}"] * len(data["mjd"]))
            else:
                raise ValueError(f"Unknown instrument: {instrument}")

    # Assemble df
    data = pd.DataFrame(
        {
            "mjd": mjd,
            "mag": mag,
            "magerr": magerr,
            "filter": filter,
        }
    )

    return data
