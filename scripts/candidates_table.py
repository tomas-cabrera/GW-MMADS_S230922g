import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP9 as cosmo
from ligo.skymap.io.fits import read_sky_map
from ligo.skymap.postprocess import crossmatch

####################################################################################################


def best_z(
    row,
    z_columns=[
        "ls_Z_SPEC",
        # "ned_z",
        "quaia_redshift_quaia",
        "ls_Z_PHOT_MEDIAN",
    ],
):
    # LS specz
    z = row["ls_Z_SPEC"]
    if not np.isnan(z) and z != -99:
        return "ls_Z_SPEC", z

    # NED z
    # z = row["ned_z"]
    # if not np.isnan(z):
    #    return "ned_z", z

    # Quaia SPz
    z = row["quaia_redshift_quaia"]
    if not np.isnan(z) and row["quaia_sep_deg"] < 1 / 3600:
        return "quaia_redshift_quaia", z

    # LS photz
    z = row["ls_Z_PHOT_MEDIAN"]
    if not np.isnan(z) and z != -99:
        return "ls_Z_PHOT_MEDIAN", z

    return "-", np.nan

    return zcol, z


def best_parsnip_classification(
    row, classes=["KN", "SLSN-I", "SNII", "SNIbc", "TDE", "SNIa"]
):
    # Get probabilities
    probs = [row[c] for c in classes]
    # Get index of maximum probability
    imax = np.argmax(probs)
    # Return class and probability
    parsnip_class = classes[imax]
    parsnip_prob = probs[imax]
    # print(imax, classes, probs)
    # print(parsnip_class, parsnip_prob)
    return parsnip_class, parsnip_prob


####################################################################################################

PATH_TO_SOURCEASSOC = "/home/tomas/academia/projects/decam_followup_O4/S230922g/data/decam/SourceASSOC.vera240103.shortlist.perlmutterpostprocessed.photoz_with_i.csv"
PATH_TO_GWSKYMAP = "/home/tomas/Data/gwemopt/skymaps/S230922g_4_Bilby.multiorder.fits,0"
PATH_TO_PARSNIP = "/home/tomas/academia/projects/decam_followup_O4/S230922g/GW-MMADS_S230922g/data/classification/vera240103.shortlist.parsnip.vdiaz.csv"

####################################################################################################

# Load data
df_sa = pd.read_csv(PATH_TO_SOURCEASSOC).set_index("OBJID")

# Select redshifts
df_sa[["table_z_source", "table_z"]] = df_sa.apply(best_z, axis=1, result_type="expand")

##############################
###    Add ParSNIP data    ###
##############################

# Load ParSNIP data
df_parsnip = pd.read_csv(PATH_TO_PARSNIP).set_index("Name")
print(df_sa.shape)
print(df_parsnip.shape)

# Select ParSNIP classification
df_parsnip[["table_parsnip_class", "table_parsnip_prob"]] = df_parsnip.apply(
    best_parsnip_classification, axis=1, result_type="expand"
)

# Merge ParSNIP data
df_sa = df_sa.join(df_parsnip[["table_parsnip_class", "table_parsnip_prob"]])
del df_parsnip

##############################
###   Calc. skymap probs.  ###
##############################

# Open skymap
skymap = read_sky_map(PATH_TO_GWSKYMAP, moc=True)

# Make SkyCoord object
ra = df_sa["RA_OBJ"].values * u.deg
dec = df_sa["DEC_OBJ"].values * u.deg
distance = cosmo.luminosity_distance(df_sa["table_z"].values)
sc_sa = SkyCoord(
    ra=ra,
    dec=dec,
    distance=distance,
)

# Crossmatch with skymap
xm_sa = crossmatch(skymap, sc_sa)
df_sa["skymap_probdensity"] = xm_sa.probdensity
df_sa["skymap_searched_prob"] = xm_sa.searched_prob
df_sa["skymap_probdensity_vol"] = xm_sa.probdensity_vol
df_sa["skymap_searched_prob_vol"] = xm_sa.searched_prob_vol

##############################
###   Generate table tex   ###
##############################

# Format columns as strings
df_sa["table_z_str"] = df_sa["table_z"].apply(
    lambda x: f"{x:.3f}" if ~np.isnan(x) else "-"
)
df_sa["table_z_source_str"] = df_sa["table_z_source"].apply(
    lambda x: {
        "ls_Z_SPEC": "LS specz",
        "ned_z": "NED z",
        "quaia_redshift_quaia": "Quaia SPz",
        "ls_Z_PHOT_MEDIAN": "LS photz",
        "-": "-",
    }[x]
)
df_sa["skymap_searched_prob_str"] = df_sa["skymap_searched_prob"].apply(
    lambda x: f"{x:.3f}"
)
df_sa["skymap_searched_prob_vol_str"] = df_sa["skymap_searched_prob_vol"].apply(
    lambda x: f"{x:.3f}" if x < 1 else "-"
)
df_sa["table_parsnip_class_str"] = df_sa["table_parsnip_class"].apply(
    lambda x: x if type(x) == str else "-"
)
df_sa["table_parsnip_prob_str"] = df_sa["table_parsnip_prob"].apply(
    lambda x: f"{x:.3f}" if ~np.isnan(x) else "-"
)

# Select columns
table_cols = [
    "table_z_str",
    "table_z_source_str",
    "skymap_searched_prob_str",
    "skymap_searched_prob_vol_str",
    "table_parsnip_class_str",
    "table_parsnip_prob_str",
]

# Sort by skymap probability
df_sa.sort_values("skymap_searched_prob", inplace=True)

# Iterate over rows
tablestr = ""
for ri, row in df_sa.iterrows():
    # Add data
    tempstr = " & ".join([ri] + [str(row[col]) for col in table_cols])
    # Add newline
    tempstr += r" \\" + "\n"
    # Add to tablestr
    tablestr += tempstr

# Write to file
texpath = __file__.replace("/scripts/", "/tex/").replace(".py", ".tex")
with open(texpath, "w") as f:
    f.write(tablestr)
