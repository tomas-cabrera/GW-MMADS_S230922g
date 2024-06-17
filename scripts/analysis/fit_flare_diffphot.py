import argparse
import math
import os.path as pa
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.cosmology import Planck18
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import curve_fit

from scripts.utils import light_curves, paths, plotting

###############################################################################


def abmag_to_specfluxdensity(abmags):
    """
    Convert AB magnitudes to spectral flux densities.

    Parameters
    ----------
    abmags : array-like
        AB magnitudes.

    Returns
    -------
    array-like
        Spectral flux densities.
    """
    return 10 ** (-0.4 * (abmags + 48.6))


def calc_total_energy(
    times,
    mag_phots,
    mag_phot_errs,
    filter,
    z,
    z_err,
    cosmo=Planck18,
):
    """Calculates the total energy emitted in a light curve, in ergs.

    Parameters
    ----------
    times : _type_
        Times of observations, in MJD.
    mag_phots : _type_
        AB magnitudes
    mag_phot_errs : _type_
        _description_
    filter : str
        String of the format f"{instrument}-{filter}"
    z : _type_
        _description_
    z_err : _type_
        _description_
    cosmo : _type_, optional
        _description_, by default astropy.cosmology.Planck15

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    inst, fil = filter.split("-")

    ### Spectral flux density time integral ###

    # Convert AB mags to spectral flux densities
    sfdensities = (mag_phots * u.ABmag).to(u.erg / u.s / u.cm**2 / u.Hz).value

    # Calculate integral via trapezoidal rule
    time_integral = np.trapz(sfdensities, (times * u.day).to(u.s).value)

    ### Bandpass wavelength integral ###

    # Load bandpass
    bandpass = light_curves.get_bandpass(filter)

    # Convert wavelength to frequency; sort by frequency
    bandpass["nu"] = (
        (bandpass["lambda"].values * u.AA).to(u.Hz, equivalencies=u.spectral()).value
    )
    argsort = np.argsort(bandpass["nu"])
    bandpass = bandpass.iloc[argsort]

    # Integrate bandpass
    bandpass_integral = np.trapz(
        bandpass["transmission"],
        bandpass["nu"],
    )

    ### Combine into total energy ###

    # Multiply integrals
    total_energy = time_integral * bandpass_integral

    # Convert flux to luminosity
    total_energy *= 4 * np.pi * (cosmo.luminosity_distance(z)).to(u.cm).value ** 2

    # Convert band luminosity to bolometric luminosity (values from Duras+20)
    # If inst == sncosmo, scale Duras+20 5.15 factor by relative transmission between filters
    if inst == "sncosmo":
        # Integrate B bandpass
        bandpass_B = light_curves.get_bandpass("sncosmo-bessellb")
        bandpass_B["nu"] = (
            (bandpass_B["lambda"].values * u.AA)
            .to(u.Hz, equivalencies=u.spectral())
            .value
        )
        argsort = np.argsort(bandpass_B["nu"])
        bandpass_B = bandpass_B.iloc[argsort]
        bandpass_B_integral = np.trapz(
            bandpass_B["transmission"],
            bandpass_B["nu"],
        )
        # Scale bolometric correction
        bolometric_correction = 5.15 * bandpass_B_integral / bandpass_integral
    # Else, just use 5.15 factor
    elif fil in list("ugrizYUBV"):
        bolometric_correction = 5.15
    else:
        raise ValueError(f"Filter [{filter}] not recognized.")
    total_energy *= bolometric_correction

    ### Error propagation ###

    # TODO: Propogate errors, returning None if None
    if mag_phot_errs is None and z_err is None:
        total_energy_err = None

    ### Return ###
    return total_energy, total_energy_err


def gaussrise_expdecay(t, t0, f0, A, tau_rise, tau_decay):
    """
    Gaussian rise, exponential decay flare model.

    Parameters
    ----------
    t : float
        Time.
    t0 : float
        Time of flare peak.
    f0 : float
        Baseline flux.
    A : float
        Amplitude of flare.
    tau_rise : float
        Rise time constant.
    tau_decay : float
        Decay time constant.

    Returns
    -------
    float
        Flux at time t.
    """
    return f0 + A * (
        np.where(t < t0, 1, 0) * np.exp(-0.5 * ((t - t0) / tau_rise) ** 2)
        + np.where(t < t0, 0, 1) * np.exp(-(t - t0) / tau_decay)
    )


def fit_flare(t, f, model, **kwargs):
    """
    Fit a flare model to a light curve.

    Parameters
    ----------
    t : array-like
        Time.
    f : array-like
        Flux.
    model : function
        Flare model.
    p0 : array-like, optional
        Initial guess for model parameters.

    Returns
    -------
    array-like
        Best-fit model parameters.
    """

    # Fit model
    popt, pcov = curve_fit(model, t, f, **kwargs)

    return popt, pcov


###############################################################################

if __name__ == "__main__":
    ##############################
    ###     Setup argparse     ###
    ##############################

    # parser object
    parser = argparse.ArgumentParser("Fit a flare model to a light curve.")

    # Model
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model to fit to the light curve.",
        default="gaussrise_expdecay",
    )

    ##############################
    ###          Setup         ###
    ##############################

    # Parse args
    args = parser.parse_args()

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

    # Define photometry directory
    DECAM_DIR = paths.PHOTOMETRY_DIR / "DECam"

    ##############################
    ###          Main          ###
    ##############################

    # Iterate over objects
    df_master = []
    for objid in candidates_table["objid"]:
        print("*" * 60)
        print(objid)

        ### Difference photometry
        # Get the first file that matches the object name
        glob_str = str(DECAM_DIR / "diffphot" / f"*{objid}.fits")
        fitsname = paths.glob_plus(glob_str, require_one=True)
        print(f"diffphot fitsname: {fitsname}")

        # Read in photometry
        with fits.open(fitsname) as hdul:
            data_phot = Table(hdul[1].data)

        # Drop nans
        data_phot = data_phot[
            ~np.isnan(data_phot["MAG_FPHOT"]) & ~np.isnan(data_phot["MJD_OBS"])
        ]

        # Remove bad expnames
        mask_is_bad_expname = np.array(
            [s in plotting.BAD_EXPNAMES for s in data_phot["SCIENCE_NAME"]]
        )
        data_phot = data_phot[~mask_is_bad_expname]

        # Initialize plot
        fig, axd = plt.subplot_mosaic(
            [
                ["LC", "LC"],
                ["LC", "LC"],
                # ["SF0", "SF1"],
            ],
        )

        # Choose model, initial params
        if args.model == "gaussrise_expdecay":
            model = gaussrise_expdecay

        # Iterate over filters
        df_params = []
        for filt, k in zip(np.unique(data_phot["FILTER"]), ["SF0", "SF1"]):
            print("*" * 10, filt, "*" * 10)

            # Initialize param dict
            dict_params = {
                "band": filt,
            }

            # Select filter data
            data_phot_filt = data_phot[(data_phot["FILTER"] == filt)]
            data_phot_filt.sort("MJD_OBS")

            # Make nondetections mask
            nondetections = data_phot_filt["STATUS_FPHOT"] == "m"

            ##############################
            ###    Plot light curve    ###
            ##############################

            axd["LC"].plot(
                data_phot_filt[~nondetections]["MJD_OBS"],
                data_phot_filt[~nondetections]["MAG_FPHOT"],
                label=f"{filt}",
                ls="",
                marker="o",
                color=plotting.band2color[filt],
                alpha=0.3,
            )
            axd["LC"].plot(
                data_phot_filt[nondetections]["MJD_OBS"],
                data_phot_filt[nondetections]["MAG_FPHOT"],
                label=f"{filt}",
                ls="",
                marker="v",
                color=plotting.band2color[filt],
                alpha=0.3,
            )

            ##############################
            ###   Calc. total energy   ###
            ##############################

            # Load redshift
            if objid in candidates_table["objid"].values:
                z = candidates_table["z"][candidates_table["objid"] == objid].values[0]
            else:
                z = np.nan
            if np.isnan(z):
                print(
                    f"Calc. total energy: Redshift not found for {objid}; setting to 0.1."
                )
                z = 0.1

            ### Preprocess light curve ###

            # Clip to etot_band
            data_phot_etot = data_phot_filt.copy()

            # Set nondetections as zero flux
            data_phot_etot["MAG_FPHOT"][nondetections] = np.inf

            # Sort by time
            argsort = np.argsort(data_phot_etot["MJD_OBS"])
            data_phot_etot = data_phot_etot[argsort]

            ### Calculate total energy ###

            total_energy, total_energy_err = calc_total_energy(
                times=data_phot_etot["MJD_OBS"],
                mag_phots=data_phot_etot["MAG_FPHOT"],
                mag_phot_errs=None,
                # filter=f"DECam-{filt}",
                filter=f"sncosmo-des{filt}",
                z=z,
                z_err=None,
            )

            # Append to dict
            dict_params.update(
                {
                    "total_energy": total_energy,
                    "total_energy_err": total_energy_err,
                }
            )
            print(f"Total energy (error): {total_energy:.2e} ({total_energy_err}) ergs")

            # Shade integral area on plot

            axd["LC"].fill_between(
                data_phot_etot["MJD_OBS"],
                data_phot_etot["MAG_FPHOT"],
                np.nanmax(data_phot_etot[~nondetections]["MAG_FPHOT"]),
                alpha=0.3,
                color=plotting.band2color[filt],
            )

            ##############################
            ###  Calc. SF probability  ###
            ##############################

            # Initialize vectorized erfc
            erfc_vec = np.vectorize(math.erfc)

            ## Dimmest before maximum, minimum sf prob
            # Get index for brightest observation
            ibright = np.nanargmin(data_phot_filt["MAG_FPHOT"])
            # Get index for dimmest observation before brightest; default to 0
            try:
                idim = np.nanargmax(
                    data_phot_filt[
                        data_phot_filt["MJD_OBS"] < data_phot_filt[ibright]["MJD_OBS"]
                    ]["MAG_FPHOT"]
                )
            except:
                idim = 0
            data_phot_filt_detections = data_phot_filt[~nondetections]
            # Get time deltas, enforcing a minimum of 1 day (to avoid deltas from the same night)
            deltats = (
                data_phot_filt_detections["MJD_OBS"] - data_phot_filt["MJD_OBS"][idim]
            )
            deltats = np.maximum(deltats, deltats / np.abs(deltats))
            # Get magnitude deltas
            deltams = (
                data_phot_filt_detections["MAG_FPHOT"]
                - data_phot_filt["MAG_FPHOT"][idim]
            )
            # Calculate SF probabilities
            sfs = (
                light_curves.KIMURA20_SF_PARAMS.loc[filt, "SF0"]
                * (deltats / light_curves.KIMURA20_SF_PARAMS.loc[filt, "dt0"])
                ** light_curves.KIMURA20_SF_PARAMS.loc[filt, "bt"]
            )
            sf_zscores = -deltams / sfs
            sf_prob = erfc_vec(sf_zscores)
            # Find minimum sf prob
            try:
                sf_prob_min = np.nanmin(sf_prob)
                ipair = np.nanargmin(sf_prob)
                deltats_opt = deltats[ipair]
                deltams_opt = deltams[ipair]
                sf_zscores_opt = sf_zscores[ipair]
            except ValueError:
                sf_prob_min = np.nan
                ipair = 0
                sf_zscores_opt = np.nan
            ipair = np.where(
                data_phot_filt["SCIENCE_NAME"]
                == data_phot_filt_detections[ipair]["SCIENCE_NAME"]
            )[0][0]
            i_sf = [idim, ipair]

            # Indicate points used on photometry plot
            axd["LC"].plot(
                data_phot_filt[i_sf]["MJD_OBS"],
                data_phot_filt[i_sf]["MAG_FPHOT"],
                label=f"{filt}",
                # ls="",
                marker="s",
                markerfacecolor="none",
                color=plotting.band2color[filt],
            )

            # Add params to dict
            dict_params.update(
                {
                    "deltams": deltams_opt,
                    "deltats": deltats_opt,
                    "ilo_sf": i_sf[0],
                    "ihi_sf": i_sf[1],
                    "sf_prob": sf_prob_min,
                    "sf_zscore": sf_zscores_opt,
                }
            )
            print(f"SF minimum probability: {sf_prob_min}")

            ##############################
            ### Fit gaussrise expdecay ###
            ##############################

            # Select filter data
            mask_filter = data_phot["FILTER"] == filt
            data_phot_filt = data_phot[mask_filter]

            # Calculate flux
            data_phot_filt["FLUX"] = light_curves.mag2fluxdensity(
                data_phot_filt["MAG_FPHOT"],
                band=filt,
            )

            # Upscale flux for numerics
            fscale = 1e24
            data_phot_filt["FLUX"] *= fscale

            # Select initial guess
            t0 = data_phot_filt["MJD_OBS"][np.argmax(data_phot_filt["FLUX"])]
            f0 = np.median(data_phot_filt["FLUX"])
            A = np.max(data_phot_filt["FLUX"]) - f0
            tau_rise = 10
            tau_decay = 10
            p0 = [t0, f0, A, tau_rise, tau_decay]

            # Fit model
            ts = data_phot_filt["MJD_OBS"]
            fs = data_phot_filt["FLUX"]
            bounds = (
                [ts.min(), 0, 0, 0, 0],
                [
                    ts.max(),
                    fs.max(),
                    fs.max(),
                    np.inf,
                    np.inf,
                ],
            )
            try:
                fit_params = fit_flare(
                    ts,
                    fs,
                    model,
                    p0=p0,
                    bounds=bounds,
                )[0]
            except Exception as e:
                print(f"ERROR FITTING MODEL: {e}")
                fit_params = [np.nan] * 5

            # Downscale flux params
            fit_params[1] /= fscale
            fit_params[2] /= fscale

            # Save to dict
            dict_params.update(
                {
                    "t0": fit_params[0],
                    "f0": fit_params[1],
                    "A": fit_params[2],
                    "tau_rise": fit_params[3],
                    "tau_decay": fit_params[4],
                }
            )
            print(
                f"Fitparams: t0={fit_params[0]:.2f},",
                f"f0={fit_params[1]:.2e},",
                f"A={fit_params[2]:.2e},",
                f"tau_rise={fit_params[3]:.2f} [log10(t_g)={np.log10(fit_params[3]):.2f}],",
                f"tau_decay={fit_params[4]:.2f} [log10(t_e)={np.log10(fit_params[4]):.2f}],",
            )

            ### Plot model ###

            # Choose time points
            tsamp = np.linspace(
                np.min(data_phot["MJD_OBS"]), np.max(data_phot["MJD_OBS"]), 100
            )

            # Plot model, print params
            axd["LC"].plot(
                tsamp,
                light_curves.fluxdensity2mag(
                    gaussrise_expdecay(tsamp, *fit_params), band=filt
                ),
                label=f"Model {filt}",
                color=plotting.band2color[filt],
                ls="--",
                alpha=0.5,
            )

            ##############################
            ###  Append to df_params   ###
            ##############################

            df_params.append(dict_params)

        axd["LC"].set_xlabel("Time")
        axd["LC"].set_ylabel("Mag")
        axd["LC"].invert_yaxis()
        # plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        # plt.show()
        plt.close()

        # Combine params and save as csv
        df_params = pd.DataFrame(df_params)
        df_params.set_index("band").to_csv(paths.output / f"{objid}_fitparams.csv")

        # Save to master df
        df_params["object"] = objid
        df_master.append(df_params)

    # Save master df
    df_master = pd.concat(df_master)
    df_master.set_index("object").to_csv(paths.output / "fitparams_master.csv")
