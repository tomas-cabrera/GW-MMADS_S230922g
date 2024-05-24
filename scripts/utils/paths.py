"""
Exposes common paths useful for manipulating datasets and generating figures.

"""

import glob
from pathlib import Path

# Absolute path to the top level of the repository
root = Path(__file__).resolve().parents[2].absolute()

# Absolute path to the `root/data` folder (contains datasets)
data = root / "data"

# Absolute path to the `root/static` folder (contains static images)
static = root / "static"

# Absolute path to the `root/scripts` folder (contains figure/pipeline scripts)
scripts = root / "scripts"

# Absolute path to the `root/tex` folder (contains the manuscript)
tex = root / "tex"

# Absolute path to the `root/tex/figures` folder (contains figure output)
figures = tex / "figures"

# Absolute path to the `root/tex/tables` folder (contains table output)
tables = tex / "tables"

# Absolute path to the `root/tex/output` folder (contains other user-defined output)
output = tex / "output"

###############################################################################

EVENTNAME = "S230922g"
SKYMAP_DIR = data / "skymaps"
SKYMAP_MULTIORDER = SKYMAP_DIR / f"{EVENTNAME}_4_Bilby.multiorder.fits,0"
SKYMAP_FLATTENED = SKYMAP_DIR / f"{EVENTNAME}_4_Bilby.flattened.fits,0"
PHOTOMETRY_DIR = data / "photometry"
SPECTRA_DIR = data / "spectra"

###############################################################################


def script_to_fig(script, key=""):
    return script.replace("scripts", "tex").replace(".py", "") + f"/{key}.pdf"


def glob_plus(globstr, require_one=False):
    # Do search
    files = glob.glob(globstr)
    # Raise error if no files found
    if not files:
        raise FileNotFoundError(f"No files found with globstr: {globstr}")
    # If require_one, raise error if more than one file found
    if require_one and len(files) > 1:
        raise FileNotFoundError(f"More than one file found with globstr: {globstr}")
    # Return
    if require_one:
        return files[0]
    return files
