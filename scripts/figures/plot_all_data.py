import os
import os.path as pa

import matplotlib.pyplot as plt

import scripts.utils.light_curves as mylc
from scripts.utils import paths, plotting

###############################################################################


def make_plot(objid, kw_subplots={"figsize": (4, 3)}):
    print("*" * 60)
    print(objid)

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
                markersize=2,
                color=plotting.band2color[f],
                capsize=1.5,
                # **kw,
            )

    ax.set_title(objid)
    ax.invert_yaxis()
    artists = {**artists_insts, **artists_fs}
    ax.set_xlabel("MJD")
    ax.set_ylabel("Magnitude")
    plt.legend(handles=artists.values(), labels=artists.keys())
    plt.tight_layout()
    # Save plot
    outpath = paths.figures / pa.basename(__file__).replace(".py", "") / f"{objid}.pdf"
    os.makedirs(pa.dirname(outpath), exist_ok=True)
    plt.savefig(outpath)
    plt.close()


###############################################################################


if __name__ == "__main__":
    make_plot("C202309242206400m275139")
