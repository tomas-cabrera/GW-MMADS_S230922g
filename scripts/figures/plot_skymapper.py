import os
import os.path as pa
import matplotlib.pyplot as plt
from scripts.utils import paths, plotting
from scripts.utils import light_curves as mylc

obj = "C202309242206400m275139"

##############################
###   All photometry plot  ###
##############################

# Select axes
fig, ax = plt.subplots(
    # figsize=(5.5, 3),
    figsize=(4, 3),
)

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
custom_legend = False
for inst, df in data_dict.items():
    if inst != "SkyMapper":
        continue

    # Add instrument marker to artist list
    if custom_legend:
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
        if custom_legend:
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
        marker = plotting.kw_instruments[inst]["marker"]
        if custom_legend:
            ax.errorbar(
                df_plot["mjd"],
                df_plot["mag"],
                yerr=df_plot["magerr"],
                ls="",
                lw=0.5,
                marker=marker,
                markersize=3,
                markeredgewidth=0,
                color=plotting.band2color[f],
                capsize=1.5,
                # **kw,
            )
        else:
            ax.errorbar(
                df_plot["mjd"],
                df_plot["mag"],
                yerr=df_plot["magerr"],
                ls="",
                lw=0.5,
                marker=marker,
                markersize=3,
                markeredgewidth=0,
                color=plotting.band2color[f],
                capsize=1.5,
                label=f"{inst} {f}",
                # **kw,
            )

ax.invert_yaxis()
ax.set_xlabel("Time (MJD)")
ax.set_ylabel("Magnitude")
artists = {**artists_insts, **artists_fs}
if custom_legend:
    ax.legend(
        handles=artists.values(),
        labels=artists.keys(),
        frameon=True,
    )
else:
    ax.legend(
        frameon=True,
    )

# Grid
ax.grid(True, which="major", color="gray", alpha=0.5, zorder=-1)
ax.set_axisbelow(True)

# Custom limits
ax.set_xlim(57860, 57990)

# Save figure
plt.tight_layout()
figpath = paths.script_to_fig(__file__, key=f"{obj}")
os.makedirs(pa.dirname(figpath), exist_ok=True)
plt.savefig(figpath)
plt.close()
