import argparse

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import curve_fit
from utils import light_curves, plotting

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

    # Photometry file
    parser.add_argument(
        "file_phot", type=str, help="Path to file containing photometry."
    )

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

    # Read in photometry
    data_phot = Table(fits.open(args.file_phot)[1].data)

    # Drop nans
    data_phot = data_phot[
        ~np.isnan(data_phot["MAG_FPHOT"]) & ~np.isnan(data_phot["MJD_OBS"])
    ]

    # Remove bad expnames
    mask_is_bad_expname = np.array(
        [s in plotting.BAD_EXPNAMES for s in data_phot["SCIENCE_NAME"]]
    )
    data_phot = data_phot[~mask_is_bad_expname]

    # Choose model, initial params
    if args.model == "gaussrise_expdecay":
        model = gaussrise_expdecay

    # Fit model filter by filter
    popts = {}
    for filt in np.unique(data_phot["FILTER"]):
        # Select filter data
        mask_filter = data_phot["FILTER"] == filt
        data_phot_filt = data_phot[mask_filter]

        # Calculate flux
        data_phot_filt["FLUX"] = light_curves.mag_to_flux(
            data_phot_filt["MAG_FPHOT"],
            band=filt,
            flux_unit=u.def_unit("miniflux", 1e-20 * u.erg / u.s / u.cm**2 / u.AA),
        )

        # Select initial guess
        t0 = data_phot_filt["MJD_OBS"][np.argmax(data_phot_filt["FLUX"])]
        f0 = np.median(data_phot_filt["FLUX"])
        A = np.max(data_phot_filt["FLUX"]) - f0
        tau_rise = 10
        tau_decay = 10
        p0 = [t0, f0, A, tau_rise, tau_decay]
        # p0 = [60250, f0, A, tau_rise, tau_decay]

        # Fit model
        ts = data_phot_filt["MJD_OBS"]
        fs = data_phot_filt["FLUX"]
        popts[filt] = fit_flare(
            ts,
            fs,
            model,
            p0=p0,
            bounds=(
                [ts.min(), 0, 0, 0, 0],
                [
                    ts.max(),
                    fs.max(),
                    fs.max(),
                    np.inf,
                    np.inf,
                ],
            ),
        )[0]

    # Initialize plot
    fig, ax = plt.subplots()  # figsize=(6, 6))

    # Plot light curve and best-fit model
    tsamp = np.linspace(np.min(data_phot["MJD_OBS"]), np.max(data_phot["MJD_OBS"]), 100)
    for filt, popt in popts.items():
        # Select filter data
        data_phot_filt = data_phot[data_phot["FILTER"] == filt]

        # Calculate flux
        data_phot_filt["FLUX"] = light_curves.mag_to_flux(
            data_phot_filt["MAG_FPHOT"],
            band=filt,
            flux_unit=u.def_unit("miniflux", 1e-20 * u.erg / u.s / u.cm**2 / u.AA),
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

        # Calculate
        ax.plot(
            tsamp,
            gaussrise_expdecay(tsamp, *popt),
            label=f"Model {filt}",
            color=plotting.band2color[filt],
        )
        print(
            f"{filt}: t0={popt[0]:.2f},",
            f"f0={popt[1]:.2f},",
            f"A={popt[2]:.2f},",
            f"tau_rise={popt[3]:.2f} [log10(t_g)={np.log10(popt[3]):.2f}],",
            f"tau_decay={popt[4]:.2f} [log10(t_e)={np.log10(popt[4]):.2f}],",
        )
    plt.xlabel("Time")
    plt.ylabel("Flux")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()
