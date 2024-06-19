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
    "C202309242223382m172348": "AT 2023uho",
    "A202310042237432m160819": "AT 2023uec",
    "A202310262246341m291842": "AT 2023aden",
    "C202309242248405m134956": "AT 2023uab",
    "C202310042207549m253435": "AT 2023unl",
    "A202309232223004m210838": "AT 2023adwb",
    "C202309242234324m242410": "AT 2023adwc",
    "C202309232219031m220609": "AT 2023adio",
    "C202311032229009m263654": "SN 2023xeh",
    "C202309242224596m265957": "AT 2023adwd",
    "A202309242229429m265636": "AT 2023adwe",
    "C202310262240474m244806": "AT 2023adwf",
    "A202310042247077m204825": "AT 2023adwg",
    "T202309242238549m260903": "AT 2023adwh",
    "C202309242248465m232916": "AT 2023adwi",
    "A202309242220316m281816": "AT 2023adwj",
    "C202309242214247m221046": "AT 2023uos",
    "A202309242212231m234356": "AT 2023adwk",
    "A202309242247422m260221": "AT 2023adwl",
    "C202309242224327m160704": "AT 2023adwm",
    "A202310042250545m160517": "AT 2023adwn",
    "T202309242209384m251125": "AT 2023uop",
    "C202309242246522m280621": "AT 2023unj",
    "T202310262246447m281410": "AT 2023adwo",
    "A202310262240047m305217": "AT 2023adwp",
    "A202309242208066m244517": "AT 2023adwq",
    "C202310042258174m224619": "AT 2023udg",
    "C202310042258328m202211": "AT 2023adwr",
    "A202312232207465m275903": "AT 2023adfo",
    "T202309242207337m283452": "AT 2019dtw",
    "A202309242207556m282406": "AT 2023adws",
    "C202311032207212m290327": "AT 2023adwt",
    "C202309242204580m282926": "AT 2023uea",
    "T202310042220383m151747": "AT 2023upf",
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
    "WISE": {"marker": "^"},
    "DECam": {"marker": "P"},
    "SkyMapper": {"marker": "X"},
    "Wendelstein": {"marker": "D"},
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
