# %% [markdown]
# # Pre-processing: derive once, stamp, verify
#
# **To open this file as a notebook** in JupyterLab: right-click it
# in the file browser and choose **Open With -> Notebook** (requires
# the `jupytext` extension in the JupyterLab environment). It also
# runs top-to-bottom as a plain Python script.
#
# The input contract (see `00_input_contract.py`) marks some
# requirements as DERIVABLE: the solar tables, the segment routing
# order, a units-conversion field, and the map-weights matrices are
# all required by the model but generatable from other supplied
# quantities. By default the assembler derives them IN-CHAIN, in
# memory, from the very arrays being supplied -- so they are always
# aligned with the run, by construction.
#
# In this notebook we take the other posture: derive them ONCE, save
# them to disk, and reuse them -- the right trade when the derivation
# inputs are fixed across many runs (say, an ensemble over forcings).
# The danger of cached artifacts is STALENESS: reusing them after the
# inputs they were derived from have changed. We'll see how the saved
# artifacts carry provenance stamps that make staleness impossible to
# miss.
#
# ## Prerequisites
#
# The `pwpx` environment and the drb_2yr test domain's legacy PRMS
# files (as in `01_prms_legacy_translation.py`).

# %%
import pathlib as pl
import sys

import numpy as np
import xarray as xr

# as a script __file__ locates us; in a Jupyter kernel the cwd is the
# notebook's directory (JupyterLab default)
try:
    _here = pl.Path(__file__).parent
except NameError:
    _here = pl.Path.cwd()
_pkg = _here.parent
assert (_pkg / "model.py").exists(), (
    f"expected to run from incarnations/mpixarray/examples, not {_here}"
)
sys.path.append(str(_pkg))

from prms_translate import write_preprocessed

MPIX_ROOT = _pkg.parents[2]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
CONTROL_FILE = DOMAIN_DIR / "nhm_stream_temp.control"

# %% [markdown]
# ## 1. Derive and save
#
# One call executes every derivation the configuration needs and
# writes the artifacts to a single NetCDF in this notebook's data
# directory.

# %%
data_dir = _here / "02_preprocessing"
data_dir.mkdir(exist_ok=True)
artifacts_path = write_preprocessed(CONTROL_FILE, data_dir / "preprocessed.nc")
artifacts = xr.load_dataset(artifacts_path)
print("artifacts:", ", ".join(str(nn) for nn in artifacts.data_vars))

# %% [markdown]
# ## 2. The provenance stamps
#
# Each artifact records HOW it was made (the derivation, straight
# from the contract) and fingerprints of WHAT it was made from -- a
# sha256 digest of each named input, plus its own digest. Let's look
# at one.

# %%
for key, val in artifacts["soltab_potsw"].attrs.items():
    print(f"{key}: {val[:64]}{'...' if len(val) > 64 else ''}")

# %% [markdown]
# ## 3. Use the artifacts
#
# Passing `preprocessed=` to the assembler (or the one-liner) uses
# the saved artifacts instead of deriving in-chain -- AFTER verifying
# every stamp against the parameters of THIS run. Because digests
# match only bit-identical arrays, what we get is provably the same
# model: let's check a couple of pieces against an in-chain assembly.

# %%
from prms_translate import assemble_from_control

kit_pre = assemble_from_control(CONTROL_FILE, preprocessed=artifacts_path)
kit_chain = assemble_from_control(CONTROL_FILE)
np.testing.assert_array_equal(
    kit_pre.process_dict["prms_atmosphere"]["parameters"][
        "soltab_potsw"
    ].values,
    kit_chain.process_dict["prms_atmosphere"]["parameters"][
        "soltab_potsw"
    ].values,
)
np.testing.assert_array_equal(
    kit_pre.maps["seg_tave_air"].weights,
    kit_chain.maps["seg_tave_air"].weights,
)
print("preprocessed and in-chain assemblies are bit-identical")

# %%
model = kit_pre.model()
model.run(np.float64(60.0 * 60.0 * 24.0), np.int32(3))
seg_tave = model.model_dict["prms_stream_temp"]["seg_tave_water"]
finite = np.isfinite(seg_tave.values)
print(
    f"3-day run from preprocessed artifacts: seg_tave_water finite "
    f"on {finite.sum()} of {finite.size} segments"
)

# %% [markdown]
# ## 4. Staleness is loud
#
# Now the point of the stamps. Suppose a calibration (or an ensemble
# member) changes `hru_slope` after the artifacts were saved -- the
# solar tables on disk no longer belong to the model being built.
# Let's simulate exactly that and verify against the perturbed
# parameters.

# %%
from prms_translate import load_parameters, verify_preprocessed

params = load_parameters(DOMAIN_DIR / "myparam.param")
perturbed = params.copy()
perturbed["hru_slope"] = perturbed["hru_slope"] * 1.0001
try:
    verify_preprocessed(artifacts, perturbed)
except ValueError as err:
    print(f"ValueError: {err}")

# %% [markdown]
# No silent wrong answers: the error names the stale artifact and the
# drifted input, and says what to do. (Note this is also why the
# in-chain default exists -- for calibration and ensembles over
# static parameters, where the derivation inputs change every run,
# skip the cache entirely and let the assembler derive in-chain.)
#
# ## Where to go next
#
# - `00_input_contract.py` shows WHERE the derivations come from: the
#   contract's derivable parameters and map weights each state theirs.
# - `01_prms_legacy_translation.py` is the in-chain default this
#   notebook is the alternative to.
