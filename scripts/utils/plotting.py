import matplotlib.pyplot as plt
import numpy as np
import tol_colors as tc
from . import paths

###############################################################################

###############################################################################

# plotting settings
plt.style.use(f"{paths.scripts}/utils/matplotlibrc.mplstyle")
plt.rc("axes", prop_cycle=plt.cycler("color", list(tc.tol_cset("bright"))))

spectra_objs = [
    "A202310262246341m291842",
    "C202309242248405m134956",
    "C202310042207549m253435",
    "C202309242206400m275139",
]
other_objs = [
    "A202310042237432m160819",
    "C202309242246522m280621",
    "C202309242214247m221046",
    "A202309242250535m165011",
    "C202309242244079m290053",
    "T202310262246447m281410",
    "C202309242204580m282926",
    "C202309242247183m132430",
    "A202309242259039m220020",
]

spectra_instruments = {
    "A202310262246341m291842": {"2023-11-16": "GN-GMOS"},
    "C202309242248405m134956": {},
    "C202310042207549m253435": {"2023-10-16": "SALT-RSS"},
    "C202309242206400m275139": {"2023-12-07": "Keck-LRIS", "2023-12-25": "GN-GMOS"},
}

# Detection type plot keywords
kw_dettag = {
    "m": {
        "marker": "v",
        "markerfacecolor": "none",
    },
    "q": {
        "marker": "o",
        # "marker": "h",
        # "markerfacecolor": "none",
    },
    "p": {
        "marker": "o",
    },
}
