import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.utils import paths, plotting

###############################################################################

# Define paths
graham23_path = paths.data / "graham23" / "graham23_fig2_posteriorsamples.csv"
fitparams_path = paths.output / "fitparams_master.csv"

# Load data
graham23 = pd.read_csv(graham23_path)
fitparams = pd.read_csv(fitparams_path)

### Plot ###
# Graham23 values
class_dict = {
    0: "Other SNe",
    1: "Ia SNe",
    2: "AGN",
    3: "TDE",
}
for c in np.unique(graham23["class"]):
    plt.hist(
        graham23.loc[graham23["class"] == c, "energy"],
        bins=100,
        # alpha=0.5,
        histtype="step",
        label=f"G23-{class_dict[c]}",
    )
# Our values
plt.vlines(
    np.log10(fitparams["total_energy"].values),
    0,
    2000,
    color="xkcd:navy",
    alpha=0.5,
    label="S230922g",
)
plt.legend()
plt.xlim(45, 55)
plt.yscale("log")
plt.xlabel("log10(Energy [erg])")
plt.ylabel("Count")
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig(__file__.replace(".py", ".png"))
plt.show()
plt.close()
