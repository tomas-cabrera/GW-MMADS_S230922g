import matplotlib.pyplot as plt

import scripts.utils.light_curves as mylc
from scripts.utils import paths, plotting

###############################################################################


def make_plot(objid, kw_subplots={"figsize": (8, 6)}):
    # Load light curves
    data_dict = {}
    for inst in plotting.kw_instruments.keys():
        try:
            df_temp = mylc.get_light_curve(objid, inst)
        except FileNotFoundError:
            continue
        data_dict[inst] = df_temp

    # Initialize plot
    fig, ax = plt.subplots(**kw_subplots)

    # Plot light curves
    for inst, df in data_dict.items():
        for f in df["filter"].unique():
            # Skip 'N/A' rows
            if f == "N/A":
                continue

            # Mask to filter
            mask = df["filter"] == f

            # Get detection type
            # dettag = df["dettag"].iloc[0]

            # Get plot keywords
            # kw = plotting.kw_dettag[dettag]

            # Plot
            ax.errorbar(
                df["mjd"][mask],
                df["mag"][mask],
                yerr=df["magerr"][mask],
                label=f"{inst} {f}",
                ls="",
                color=plotting.band2color[f],
                # **kw,
            )

    ax.invert_yaxis()
    plt.legend()
    plt.show()
    plt.close()


###############################################################################


if __name__ == "__main__":
    make_plot("C202309242206400m275139")
    make_plot("A202309242250535m165011")
