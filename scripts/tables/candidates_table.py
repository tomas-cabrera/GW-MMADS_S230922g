import astropy.units as u
from astropy.io import fits
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP9 as cosmo
from ligo.skymap.io.fits import read_sky_map
from ligo.skymap.postprocess import crossmatch
from scripts.utils import paths, plotting

####################################################################################################


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


def fast_rise(objid, cutoff=0.2 / 7):
    # Get fits
    globstr = str(paths.PHOTOMETRY_DIR / "DECam" / "diffphot" / f"*_{objid}.fits")
    fits_path = paths.glob_plus(globstr, require_one=True)
    with fits.open(fits_path) as hdul:
        # Get data, sort
        data = hdul[1].data
        data = data[np.argsort(data["MJD_OBS"])]
        # Iterate over filters
        fils = np.unique(data["FILTER"])
        for fil in fils:
            # Define mask, get data
            mask = data["FILTER"] == fil
            data_temp = data[mask]
            # Calculate deltats, deltamags
            deltats = np.diff(data_temp["MJD_OBS"])
            deltamags = np.diff(data_temp["MAG_FPHOT"])
            # Determine if there is a fast rise
            fast_rise = np.any(deltamags / deltats > cutoff)
            # Stop if fast_rise is found
            if fast_rise:
                break
    return fast_rise


def gets_bluer(objid, cutoff=0.0):
    # Get fits
    globstr = str(paths.PHOTOMETRY_DIR / "DECam" / "diffphot" / f"*_{objid}.fits")
    fits_path = paths.glob_plus(globstr, require_one=True)
    with fits.open(fits_path) as hdul:
        # Get data, sort
        data = hdul[1].data
        data = data[np.argsort(data["MJD_OBS"])]

        # Interpolate iband magnitudes at gband times
        gmask = data["FILTER"] == "g"
        imask = data["FILTER"] == "i"
        imags = np.interp(
            data["MJD_OBS"][gmask], data["MJD_OBS"][imask], data["MAG_FPHOT"][imask]
        )

        # Calculate deltamags
        deltagmi = np.diff(data["MAG_FPHOT"][gmask] - imags)
        gets_bluer = np.any(deltagmi > cutoff)

    return gets_bluer


####################################################################################################

PATH_TO_SOURCEASSOC = "/home/tomas/academia/projects/decam_followup_O4/S230922g/data/photometry/decam/SourceASSOC.vera240103.shortlist.perlmutterpostprocessed.photoz_with_i.csv"
PATH_TO_GWSKYMAP = "/home/tomas/Data/gwemopt/skymaps/S230922g_4_Bilby.multiorder.fits,0"
PATH_TO_PARSNIP = "/home/tomas/academia/projects/decam_followup_O4/S230922g/GW-MMADS_S230922g/data/classification/vera240103.shortlist.parsnip.vdiaz.csv"

INTERNAL_ZS = {
    "C202309242206400m275139": (0.184, 3.627e-5),
    "C202309242248405m134956": (0.128, 3.757e-5),
    "C202310042207549m253435": (0.248, 0.001),
}

NONCANDIDATES_POSTPROCESSING = [
    "T202309242242458m241023",
    "T202309242242259m162742",  # fast transient, no tailing nondetection
    "T202310032224474m305703",  # "kilonova", tailing nondetection
    "T202309242207149m242500",
    "T202309242258081m201140",
    "C202311032229009m263654",  # TNS classified as SN
    "T202310042213349m303327",  # Non-nuclear (see LS/PanSTARRS images)
]

####################################################################################################

# Load data
df_sa = pd.read_csv(PATH_TO_SOURCEASSOC).set_index("OBJID")

# Add internal redshifts
for objid, (z, z_err) in INTERNAL_ZS.items():
    df_sa.loc[objid, "host_z"] = z
    df_sa.loc[objid, "host_z_err"] = np.nan  # z_err
    df_sa.loc[objid, "host_catalog"] = "internal"

##############################
###    Add ParSNIP data    ###
##############################

# Load ParSNIP data
df_parsnip = pd.read_csv(PATH_TO_PARSNIP).set_index("Name")

# Select ParSNIP classification
df_parsnip[["table_parsnip_class", "table_parsnip_prob"]] = df_parsnip.apply(
    best_parsnip_classification, axis=1, result_type="expand"
)

# Merge ParSNIP data
df_sa = df_sa.join(df_parsnip[["table_parsnip_class", "table_parsnip_prob"]])
del df_parsnip

# Remove classifications without redshifts
mask = df_sa["host_z"].isna()
df_sa.loc[mask, ["table_parsnip_class", "table_parsnip_prob"]] = np.nan

##############################
###   Calc. skymap probs.  ###
##############################

# Open skymap
skymap = read_sky_map(PATH_TO_GWSKYMAP, moc=True)

# Make SkyCoord object
ra = df_sa["RA_OBJ"].values * u.deg
dec = df_sa["DEC_OBJ"].values * u.deg
distance = cosmo.luminosity_distance(df_sa["host_z"].values)
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
df_sa["table_z_str"] = df_sa["host_z"].apply(
    lambda x: f"{x:.3f}" if ~np.isnan(x) else "-"
)
df_sa["table_z_err_str"] = df_sa["host_z_err"].apply(
    lambda x: f"{x:.3f}" if ~np.isnan(x) else "-"
)
df_sa["table_z_source_str"] = df_sa["host_catalog"].apply(
    lambda x: {
        "lsdr10_specz": "LS specz",
        "ned_specz": "NED specz",
        "desi": "DESI specz",
        "quaia": "Quaia SPz",
        "lsdr10_photz": "LS photz",
        "internal": "Specz (this work)",
        np.nan: "-",
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

# Mask DESI redshifts
mask = df_sa["host_catalog"] == "desi"
df_sa.loc[mask, ["table_z_str", "table_z_err_str"]] = "*"

# Set parsnip classification to \dagger if it failed
mask = (df_sa["table_parsnip_class_str"] == "SNIa") & (
    df_sa["table_parsnip_prob"] == 0.4228443177
)
df_sa.loc[mask, ["table_parsnip_class_str", "table_parsnip_prob_str"]] = r"\dagger"

##############################
###      Perform cuts      ###
##############################

### Remove rows for objects that do not brighten by at least 0.2 mag/week at some point
mask = np.array([fast_rise(i) for i in df_sa.index])

### Remove rows for objects that do not get bluer
mask = mask & np.array([gets_bluer(i) for i in df_sa.index])

### Remove rows with high skymap_z_zscores
# (inverting the mask ensures that all rows with nan zs are kept)
mask = mask & ~(np.abs(df_sa["skymap_z_zscore"]) > 3)

### Remove noncandidates
mask = mask & ~df_sa.index.isin(NONCANDIDATES_POSTPROCESSING)

### Apply mask
df_sa = df_sa[mask]

# Select columns
table_cols = [
    "table_z_str",
    "table_z_err_str",
    "table_z_source_str",
    "skymap_searched_prob_str",
    "skymap_searched_prob_vol_str",
    "table_parsnip_class_str",
    "table_parsnip_prob_str",
]

# Sort by skymap probability
df_sa.sort_values("skymap_searched_prob", inplace=True)

##############################
###      Generate tex      ###
##############################

# Iterate over rows
tablestr = f"""\\startlongtable
\\begin{{deluxetable*}}{{cccccccc}}
    \\label{{tab:candidates}}
    \\tablecaption{{
        Summary table for our counterpart candidate shortlist.
        Redshifts are shown as available from crossmatching with several extragalactic databases and direct measurement for the objects which we took spectra (as DESI redshifts are proprietary, they are masked from the table with an ``*").
        The objects are sorted by ascending 2D skymap probability, s.t. the objects in the highest probability regions are listed first.
        The highest probability ParSNIP photometric classification along with the probability are listed in the last two columns.
    }}
    \\tablehead{{
        \\colhead{{Object}} & \\multicolumn{{3}}{{c}}{{Redshift}} & \\multicolumn{{2}}{{c}}{{GW skymap prob.}} & \\multicolumn{{2}}{{c}}{{\\colhead{{ParSNIP}}}} \\\\
        & \\colhead{{$z$}} & \\colhead{{$z_{{\\rm err}}$}} & \\colhead{{$z$ source}} & \\colhead{{2D}} & \\colhead{{3D}} & \\colhead{{Classification}} & \\colhead{{Prob.}}
    }}
    \\startdata
"""
for ri, row in df_sa.iterrows():
    # Add data
    tempstr = " & ".join(
        [plotting.tns_names[ri]] + [str(row[col]) for col in table_cols]
    )
    # Add newline
    tempstr += r" \\" + "\n"
    # Add to tablestr
    tablestr += tempstr
tablestr += f"""\enddata
\end{{deluxetable*}}"""

# Write to file
texpath = __file__.replace("/scripts/", "/tex/").replace(".py", ".tex")
with open(texpath, "w") as f:
    f.write(tablestr)
