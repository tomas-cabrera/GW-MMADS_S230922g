import matplotlib.pyplot as plt
import numpy as np
import tol_colors as tc

from scripts.utils import paths

###############################################################################

spectra_objs = [
    # "A202310262246341m291842",
    # "C202309242248405m134956",
    # "C202310042207549m253435",
    "C202309242206400m275139",
]
other_objs = [
    "C202309242206400m275139",
    "C202309242244079m290053",
    "C202309232219031m220609",
    "A202309242207556m282406",
    "C202309242209121m304249",
    "C202311032229009m263654",
    "A202309242220316m281816",
    "C202309242223382m172348",
    "C202310262240474m244806",
    "A202312232207465m275903",
    "A202310042237432m160819",
    "A202310042250545m160517",
    "A202310042246226m163533",
    "C202312022234226m204100",
    "C202312022231125m264717",
    "C202310042259300m204720",
    # "A202310042237432m160819",
    # "C202309242246522m280621",
    # "C202309242214247m221046",
    # "A202309242250535m165011",
    # "C202309242244079m290053",
    # "T202310262246447m281410",
    # "C202309242204580m282926",
    # "C202309242247183m132430",
    # "A202309242259039m220020",
    # "T202309242242259m162742",
    # "T202309242258081m201140",
    # "C202309242223382m172348",
    # "T202310032224474m305703",
]

tns_names = {
    "C202309242206400m275139": "AT 2023aagj",
    "C202309242244079m290053": "AT 2023ued",
    "C202309232219031m220609": "AT 2023",
    "A202309242207556m282406": "AT 2023",
    "C202309242209121m304249": "AT 2023",
    "C202311032229009m263654": "AT 2023",
    "A202309242220316m281816": "AT 2023",
    "C202309242223382m172348": "AT 2023uho",
    "C202310262240474m244806": "AT 2023",
    "A202312232207465m275903": "AT 2023",
    "A202310042237432m160819": "AT 2023uec",
    "A202310042250545m160517": "AT 2023",
    "A202310042246226m163533": "AT 2023",
    "C202312022234226m204100": "AT 2023",
    "C202312022231125m264717": "AT 2023",
    "C202310042259300m204720": "AT 2023",
    "A202310262246341m291842": "AT 2023aden",
    "C202309242248405m134956": "AT 2023uab",
    "C202310042207549m253435": "AT 2023unl",
}

spectra_instruments = {
    "A202310262246341m291842": {"2023-11-16": "GN-GMOS"},
    "C202309242248405m134956": {},
    "C202310042207549m253435": {"2023-10-16": "SALT-RSS"},
    "C202309242206400m275139": {"2023-12-07": "Keck-LRIS", "2023-12-25": "GN-GMOS"},
}

tol_bright = list(tc.tol_cset("bright"))
tol_dark = list(tc.tol_cset("dark"))
band2color = {
    "g": tol_bright[2],
    "r": tol_bright[1],
    "i": tol_bright[3],
    "z": tol_bright[5],
    "W1": tol_dark[3],
    "W2": tol_dark[1],
}

kw_instruments = {
    "DECam": {"marker": "h"},
    "SkyMapper": {"marker": "s"},
    "Wendelstein": {"marker": "D"},
    "WISE": {"marker": "X"},
}

# Detection type plot keywords
kw_dettag = {
    "m": {
        "marker": "v",
        "markerfacecolor": "none",
        "markersize": 5,
    },
    "q": {
        "marker": "o",
        # "marker": "h",
        # "markerfacecolor": "none",
        "markersize": 5,
    },
    "p": {
        "marker": "o",
        "markersize": 5,
    },
}

# Hard-code bad exposures
BAD_EXPNAMES = [
    "c4d_231018_004143_xxx_i_desirt",
]

###############################################################################


def candname_to_radec(candname):
    # Parse RA
    ra_str = candname[9:16]
    ra_decdeg = (
        (float(ra_str[:2]) + float(ra_str[2:4]) / 60 + float(ra_str[4:]) / 10 / 3600)
        * 360
        / 24
    )

    # Parse dec
    dec_str = candname[16:]
    if dec_str[0] == "p":
        sign = 1
    elif dec_str[0] == "m":
        sign = -1
    dec_decdeg = sign * (
        float(dec_str[1:3]) + float(dec_str[3:5]) / 60 + float(dec_str[5:]) / 10 / 3600
    )

    return ra_decdeg, dec_decdeg


###############################################################################

# plotting settings
plt.style.use(f"{paths.scripts}/utils/matplotlibrc.mplstyle")
plt.rc("axes", prop_cycle=plt.cycler("color", list(tc.tol_cset("bright"))))
