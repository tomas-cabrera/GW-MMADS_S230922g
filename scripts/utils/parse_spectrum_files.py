import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import paths

specdir = str(paths.data / "spectra")

path2colnames = {
    f"{specdir}/A202310262246341m291842_2023-11-16.ascii": [
        "wavelength",
        "flux",
        "flux_err",
    ],
    f"{specdir}/C202309242206400m275139_2023-12-07.ascii": [
        "wavelength",
        "flux",  # (erf/cm^2/s/Ang)",
        "sky_flux",
        "flux_err",
        "xpixel",
        "ypixel",
        "response",
        "flag",
    ],
    f"{specdir}/C202309242206400m275139_2023-12-25.ascii": [
        "wavelength",
        "flux",
        "flux_err",
    ],
    f"{specdir}/C202309242248405m134956_2023-10-13.ascii": [
        "wavelength",
        "flux",  # (erf/cm^2/s/Ang)",
    ],
    f"{specdir}/C202310042207549m253435_2023-10-16.ascii": [
        "wavelength",
        "flux",
        "flag",
        "something",
    ],
}

for specpath, colnames in path2colnames.items():
    # Load data
    data = pd.read_csv(specpath, delim_whitespace=True, comment="#", names=colnames)

    # Plot
    plt.plot(data["wavelength"], data["flux"] / data["flux"].median())
    plt.close()

    # Normalize flux
    data["flux"] = data["flux"] / data["flux"].median()

    # Save
    data[["wavelength", "flux"]].to_csv(
        specpath.replace(".ascii", "_parsed.csv"), index=False
    )
