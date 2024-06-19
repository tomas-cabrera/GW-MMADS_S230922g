import argparse
import math
import os
import os.path as pa
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

from scripts.utils import light_curves, paths, plotting

###############################################################################


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

    DECAM_DIR = paths.PHOTOMETRY_DIR / "DECam"
    DECAM_DIR = Path(
        "/home/tomas/academia/projects/decam_followup_O4/S230922g/data/photometry/decam/2024-01-03_shortlist/Candidates"
    )
    DIRECTPHOT_DIR = Path(
        "/home/tomas/academia/projects/decam_followup_O4/S230922g/data/photometry/decam/2024-01-03_shortlist/Candidates_directphot"
    )
    objs = os.listdir(DECAM_DIR)
    objs = [o.split("_")[-1].split(".")[0] for o in objs]
    df_master = []
    start = True
    for obj in objs:
        # Jump forward to object
        if not start:
            if obj != "A202309242245227m262607":
                continue
            else:
                start = True

        ### Difference photometry
        # Get the first file that matches the object name
        glob_str = str(DECAM_DIR / f"*{obj}.fits")
        fitsname = glob(glob_str)[0]
        print(fitsname)

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

        ### Direct photometry
        # Get the first file that matches the object name
        glob_str = str(DIRECTPHOT_DIR / f"*{obj}.fits")
        fitsname = glob(glob_str)[0]
        # print(fitsname)

        # Read in photometry
        with fits.open(fitsname) as hdul:
            data_directphot = Table(hdul[1].data)

        # Drop nans
        data_directphot = data_directphot[
            ~np.isnan(data_directphot["MAG_FPHOT"])
            & ~np.isnan(data_directphot["MJD_OBS"])
        ]

        # Remove bad expnames
        mask_is_bad_expname = np.array(
            [s in plotting.BAD_EXPNAMES for s in data_directphot["SCIENCE_NAME"]]
        )
        data_directphot = data_directphot[~mask_is_bad_expname]

        for dfp in data_phot["SCIENCE_NAME"]:
            if dfp not in data_directphot["SCIENCE_NAME"]:
                print(f"\t{dfp[0]} NOT IN DIRECTPHOT")

        ##############################
        ###  Calc. SF probability  ###
        ##############################

        # Initialize plot
        plt.close()
        handles = [
            Line2D(
                [0],
                [0],
                color="k",
                marker="o",
                alpha=0.3,
                label="Direct",
            ),
            Line2D(
                [0],
                [0],
                color="k",
                ls=":",
                marker="o",
                markerfacecolor="none",
                alpha=0.3,
                label="Diff",
            ),
            Line2D(
                [0],
                [0],
                color="k",
                marker="X",
                markerfacecolor="none",
                label="SF",
            ),
        ]
        fig, axd = plt.subplot_mosaic(
            [
                ["LC", "LC"],
                ["LC", "LC"],
                # ["SF0", "SF1"],
            ],
        )

        # Iterate over filters
        df_sfparams = []
        for filt, k in zip(np.unique(data_phot["FILTER"]), ["SF0", "SF1"]):
            # Select filter data
            data_phot_filt = data_phot[(data_phot["FILTER"] == filt)]
            data_phot_filt.sort("MJD_OBS")
            data_directphot_filt = data_directphot[(data_directphot["FILTER"] == filt)]
            data_directphot_filt.sort("MJD_OBS")

            # Plot photometry
            nondetections = data_phot_filt["STATUS_FPHOT"] == "m"
            axd["LC"].plot(
                data_phot_filt[~nondetections]["MJD_OBS"],
                data_phot_filt[~nondetections]["MAG_FPHOT"],
                # label=f"{filt}",
                ls=":",
                marker="o",
                markerfacecolor="none",
                color=plotting.band2color[filt],
                alpha=0.3,
            )
            axd["LC"].plot(
                data_phot_filt[nondetections]["MJD_OBS"],
                data_phot_filt[nondetections]["MAG_FPHOT"],
                # label=f"{filt}",
                ls="",
                marker="v",
                markerfacecolor="none",
                color=plotting.band2color[filt],
                alpha=0.3,
            )
            nondetections_directphot = data_directphot_filt["STATUS_FPHOT"] == "m"
            axd["LC"].plot(
                data_directphot_filt[~nondetections_directphot]["MJD_OBS"],
                data_directphot_filt[~nondetections_directphot]["MAG_FPHOT"],
                # label=f"{filt}",
                # ls=":",
                marker="o",
                # markerfacecolor="none",
                color=plotting.band2color[filt],
                alpha=0.3,
            )
            axd["LC"].plot(
                data_directphot_filt[nondetections_directphot]["MJD_OBS"],
                data_directphot_filt[nondetections_directphot]["MAG_FPHOT"],
                # label=f"{filt}",
                ls="",
                marker="v",
                # markerfacecolor="none",
                color=plotting.band2color[filt],
                alpha=0.3,
            )

            ### Calculate structure function probabilities
            def erfc_half(x):
                return math.erfc(x) / 2

            erfc_vec = np.vectorize(erfc_half)

            # ## Dimmest before maximum, minimum sf prob
            # # Get index for brightest observation
            # ibright = np.nanargmin(data_phot_filt["MAG_FPHOT"])
            # # Get index for dimmest observation before brightest; default to 0
            # try:
            #     idim = np.nanargmax(
            #         data_phot_filt[
            #             data_phot_filt["MJD_OBS"]
            #             < data_phot_filt[ibright]["MJD_OBS"]
            #         ]["MAG_FPHOT"]
            #     )
            # except:
            #     idim = 0
            # data_phot_filt_detections = data_phot_filt[~nondetections]
            # # Get time deltas, enforcing a minimum of 1 day (to avoid deltas from the same night)
            # deltats = (
            #     data_phot_filt_detections["MJD_OBS"]
            #     - data_phot_filt["MJD_OBS"][idim]
            # )
            # deltats = np.maximum(deltats, deltats / np.abs(deltats))
            # # Get magnitude deltas
            # deltams = (
            #     data_phot_filt_detections["MAG_FPHOT"]
            #     - data_phot_filt["MAG_FPHOT"][idim]
            # )
            # # Calculate SF probabilities
            # sfs = (
            #     light_curves.KIMURA20_SF_PARAMS.loc[filt, "SF0"]
            #     * (deltats / light_curves.KIMURA20_SF_PARAMS.loc[filt, "dt0"])
            #     ** light_curves.KIMURA20_SF_PARAMS.loc[filt, "bt"]
            # )
            # sf_zscores = -deltams / sfs
            # sf_prob = erfc_vec(sf_zscores)
            # # Find minimum sf prob
            # # print(sf_prob)
            # try:
            #     sf_prob_min = np.nanmin(sf_prob)
            #     ipair = np.nanargmin(sf_prob)
            #     sf_zscores_opt = sf_zscores[ipair]
            # except ValueError:
            #     sf_prob_min = np.nan
            #     ipair = 0
            #     sf_zscores_opt = np.nan
            # ipair = np.where(
            #     data_phot_filt["SCIENCE_NAME"]
            #     == data_phot_filt_detections[ipair]["SCIENCE_NAME"]
            # )[0][0]
            # imin = [idim, ipair]
            # # print(imin)
            # ## Calculate sf_prob for directphot
            # idim_direct = np.where(
            #     data_directphot_filt["SCIENCE_NAME"]
            #     == data_phot_filt[idim]["SCIENCE_NAME"]
            # )
            # ipair_direct = np.where(
            #     data_directphot_filt["SCIENCE_NAME"]
            #     == data_phot_filt[ipair]["SCIENCE_NAME"]
            # )
            # deltat_direct = (
            #     data_directphot_filt[ipair]["MJD_OBS"]
            #     - data_directphot_filt[idim]["MJD_OBS"]
            # )
            # deltam_direct = (
            #     data_directphot_filt[ipair]["MAG_FPHOT"]
            #     - data_directphot_filt[idim]["MAG_FPHOT"]
            # )
            # sf_direct = (
            #     light_curves.KIMURA20_SF_PARAMS.loc[filt, "SF0"]
            #     * (deltat_direct / light_curves.KIMURA20_SF_PARAMS.loc[filt, "dt0"])
            #     ** light_curves.KIMURA20_SF_PARAMS.loc[filt, "bt"]
            # )
            # sf_zscore_direct = -deltam_direct / sf_direct
            # sf_prob_direct = erfc_half(sf_zscore_direct)
            # print("\tSF_PROB:", sf_prob_min, sf_prob_direct)

            ## Maximum sf prob
            # Calculate deltas
            deltats = np.subtract.outer(
                data_directphot_filt["MJD_OBS"], data_directphot_filt["MJD_OBS"]
            )
            deltams = np.subtract.outer(
                data_directphot_filt["MAG_FPHOT"], data_directphot_filt["MAG_FPHOT"]
            )
            # Ignore deltats <= 1
            mask = deltats > 1
            deltats[~mask] = np.nan
            deltams[~mask] = np.nan
            # Calculate structure functions
            sfs = (
                light_curves.KIMURA20_SF_PARAMS.loc[filt, "SF0"]
                * (deltats / light_curves.KIMURA20_SF_PARAMS.loc[filt, "dt0"])
                ** light_curves.KIMURA20_SF_PARAMS.loc[filt, "bt"]
            )
            sf_frac = -deltams / sfs
            sf_prob = erfc_vec(sf_frac)
            # try:
            imin = np.nanargmin(sf_prob)
            # except ValueError:
            #     imin = 0
            imin = list(np.unravel_index(imin, sf_prob.shape))
            sf_prob_min = sf_prob[*imin]
            sf_zscores_opt = sf_frac[*imin]
            # print(filt, sf_zscores_opt, sf_prob_min)

            # Plot photometry
            sigma = axd["LC"].plot(
                # data_phot_filt[imin]["MJD_OBS"],
                # data_phot_filt[imin]["MAG_FPHOT"],
                data_directphot_filt[imin]["MJD_OBS"],
                data_directphot_filt[imin]["MAG_FPHOT"],
                label=f"{sf_zscores_opt:.2f}$\sigma$",
                # ls="",
                marker="X",
                markerfacecolor="none",
                color=plotting.band2color[filt],
            )
            handles.append(sigma[0])

            # Add params to df
            df_sfparams.append(
                {
                    "band": filt,
                    "ilo_sf": imin[0],
                    "ihi_sf": imin[1],
                    "sf_prob": sf_prob_min,
                    "sf_zscore": sf_zscores_opt,
                }
            )

        axd["LC"].set_xlabel("Time")
        axd["LC"].set_ylabel("Mag")
        axd["LC"].invert_yaxis()
        axd["LC"].set_title(obj)
        plt.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        # plt.show()
        plt.savefig(
            paths.figures
            / "fit_flare_directphot"
            / pa.basename(__file__).replace(".py", f"_{obj}.png")
        )
        plt.close()

        ##############################
        ### Fit gaussrise expdecay ###
        ##############################

        # Choose model, initial params
        if args.model == "gaussrise_expdecay":
            model = gaussrise_expdecay

        # Fit model filter by filter
        popts = {}
        df_fitparams = []
        for filt in np.unique(data_phot["FILTER"]):
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
            # print(bounds)
            try:
                fit_params = fit_flare(
                    ts,
                    fs,
                    model,
                    p0=p0,
                    bounds=bounds,
                )[0]
            except Exception as e:
                print(f"ERROR FITTING {obj}:{filt}; ERROR: {e}")
                fit_params = [np.nan] * 5

            # Downscale flux params
            fit_params[1] /= fscale
            fit_params[2] /= fscale

            # Save to dict
            popts[filt] = fit_params
            df_fitparams.append(
                {
                    "band": filt,
                    "t0": fit_params[0],
                    "f0": fit_params[1],
                    "A": fit_params[2],
                    "tau_rise": fit_params[3],
                    "tau_decay": fit_params[4],
                }
            )

        # Initialize plot
        fig, ax = plt.subplots()  # figsize=(6, 6))

        # Plot light curve and best-fit model
        tsamp = np.linspace(
            np.min(data_phot["MJD_OBS"]), np.max(data_phot["MJD_OBS"]), 100
        )
        for filt, popt in popts.items():
            # Select filter data
            data_phot_filt = data_phot[data_phot["FILTER"] == filt]

            # Calculate flux
            data_phot_filt["FLUX"] = light_curves.mag2fluxdensity(
                data_phot_filt["MAG_FPHOT"],
                band=filt,
            )

            # Plot photometry
            ax.plot(
                data_phot_filt["MJD_OBS"],
                data_phot_filt["FLUX"],
                label=f"Data {filt}",
                ls="",
                marker="o",
                color=plotting.band2color[filt],
            )

            # Plot model, print params
            ax.plot(
                tsamp,
                gaussrise_expdecay(tsamp, *popt),
                label=f"Model {filt}",
                color=plotting.band2color[filt],
            )
            print(
                f"{filt}: t0={popt[0]:.2f},",
                f"f0={popt[1]:.2e},",
                f"A={popt[2]:.2e},",
                f"tau_rise={popt[3]:.2f} [log10(t_g)={np.log10(popt[3]):.2f}],",
                f"tau_decay={popt[4]:.2f} [log10(t_e)={np.log10(popt[4]):.2f}],",
            )

            ### Calculate structure function probability
            # Get maximum deltamag/deltat

            # Calculate SF probability

            # Add to legend

        plt.xlabel("Time")
        plt.ylabel("Flux")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        # plt.show()
        plt.close()

        # Combine params and save as csv
        df_fitparams = pd.DataFrame(df_fitparams).set_index("band")
        df_sfparams = pd.DataFrame(df_sfparams).set_index("band")
        df_params = pd.concat([df_fitparams, df_sfparams], axis=1)
        # df_params.to_csv(paths.output / f"{c}_fitparams.csv")

        # Save to master df
        df_params["object"] = obj
        df_master.append(df_params)

    # Save master df
    df_master = pd.concat(df_master)
    # df_master.to_csv(paths.output / "fitparams_master.csv")
