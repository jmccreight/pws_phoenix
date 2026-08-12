<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [pyPRMS upstream asks](#pyprms-upstream-asks)
  - [1. Type-F parameter text is truncated to float32 at parse](#1-type-f-parameter-text-is-truncated-to-float32-at-parse)
  - [2. `force_default` makes the dynamic-parameter pathnames unreachable](#2-force_default-makes-the-dynamic-parameter-pathnames-unreachable)
  - [3. GSFLOW agricultural (soilzone_ag) metadata is missing](#3-gsflow-agricultural-soilzone_ag-metadata-is-missing)
  - [4a. Multi-valued `param_file` truncates every path to one character](#4a-multi-valued-param_file-truncates-every-path-to-one-character)
  - [4b. Partial parameter files parse as empty (silently)](#4b-partial-parameter-files-parse-as-empty-silently)
  - [5. `Control.dynamic_parameters` returns prose for some flags](#5-controldynamic_parameters-returns-prose-for-some-flags)
  - [6. Feature ask: a dynamic-parameter file reader](#6-feature-ask-a-dynamic-parameter-file-reader)
- [Migration map (the pyPRMS split-out boundary)](#migration-map-the-pyprms-split-out-boundary)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# pyPRMS upstream asks

Issue drafts for github.com/paknorton/pyPRMS, distilled from the
workarounds this package carries (see `readers.py`'s module
docstring). Each ask is demonstrated by a **feature branch in the
local pyPRMS clone** (`mpix/pyPRMS`, branched off `development` at
v0.10.0 / e4806aa) carrying a test in pyPRMS's own test style --
`xfail(strict=True)` where the branch demonstrates a defect (the
marker errors loudly the day the defect is fixed), green where the
branch includes the fix. Run any branch's suite with:

    cd mpix/pyPRMS && git switch <branch>
    PYTHONPATH=$PWD pytest tests/

Common context for every issue: we are building a PRMS -> pws_phoenix
translation layer that leans on pyPRMS as its only decoder of legacy
PRMS/GSFLOW files (control, parameter, CBH), with bitwise-parity
standards downstream. We are happy to turn any of these branches into
a PR.

---

## 1. Type-F parameter text is truncated to float32 at parse

**Branch:** `ask/f64-parameter-parse` (test-only, xfail) --
`tests/func/test_ParameterFile.py::test_type_f_parameter_full_precision`

ParameterFile parses type-F values at float32 **at the text line**
(`constants.PTYPE_TO_DTYPE[2]`, selected by the type code in the file
itself); the metadata datatype only recasts afterward, so ~1e-9
relative truncation is baked in and the full decimal precision
written in the file is unreachable:

```python
pdb = ParameterFile('myparam.param', metadata=MetaData().metadata)
vals = np.asarray(pdb.get('gwflow_coef').data, dtype=np.float64)
vals[0] == 0.072118  # False: 0.07211800000071526
```

This matters for model-parity work and calibration round-trips.
Verified workaround: patch `PTYPE_TO_DTYPE[2] = np.float64` **and**
widen the metadata datatype declarations float32 -> float64 (the
Parameter data setter re-truncates otherwise) -- this gives exact
equality with an independent float64 parse. Suggested fix shapes:
parse at the metadata-declared dtype, or a ParameterFile dtype
option.

## 2. `force_default` makes the dynamic-parameter pathnames unreachable

**Branch:** `ask/dynamic-path-force-default` (test-only, xfail) --
`tests/func/test_ControlFile.py::test_dynamic_parameter_paths_from_file`

All 19 `*_dynamic` control entries (the dynamic-parameter file
pathnames) carry `force_default` in `control.xml`, so
`ControlVariable.values` returns the metadata default regardless of
the pathname actually written in the control file:

```python
# control file contains: springfrost_dynamic = spring_frost.dyn
ctl = ControlFile('control.dyn_paths', metadata=MetaData().metadata)
ctl.get('springfrost_dynamic').values  # 'dyn_spring_frost.param' (!)
```

Framed as a question: the force_default mechanism is documented and
presumably intentional, but applying it to user-specifiable pathnames
silently yields wrong paths when reading real controls. If
intentional, what is the intended way to recover the file's actual
value? Our workaround strips force_default from the `*_dynamic`
entries before reading.

## 3. GSFLOW agricultural (soilzone_ag) metadata is missing

**Branch:** `ask/ag-gsflow-metadata` (**fix included**, suite green:
xml additions + tests + fixtures; `ctl_metadata_default.csv`
regenerated)

pyPRMS's metadata predates the GSFLOW agricultural extensions, and
each reader handles unknown names differently:

- **ParameterFile silently SKIPS** unknown parameters: a GSFLOW
  parameter file loses all 13 ag parameters (`ag_frac`,
  `ag_soil_*`, `ag_cov*`, `ag_soil2gw_max`,
  `ag_soilwater_deficit_min`, `max_soilzone_ag_iter`,
  `soilzone_aet_converge`) with no warning.
- **ControlFile RAISES** on the first unknown entry
  (`iter_aet_flag`), so a GSFLOW agricultural control file cannot
  load at all (8 entries missing: iter_aet_flag,
  forcing_check_flag, dyn_ag_frac_flag, ag_frac_dynamic,
  AET_cbh_file, PET_cbh_file, AET_module, PET_ag_module).
- **Cbh raises KeyError** on the OpenET `actet` CBH variable.

The branch adds all of it (entries tagged `version="5.2"`;
`dyn_ag_frac_flag`'s valid_values names `ag_frac` so
`Control.dynamic_parameters` reports it). NOTE for review:
minimum/maximum/default of the parameter entries mirror the
analogous non-ag parameters and should be checked against the GSFLOW
soilzone_ag declarations. A secondary observation worth its own
thought: the skip-vs-raise asymmetry between the readers (silent
data loss vs hard failure) -- and/or a supported way for users to
extend the metadata without patching package internals.

## 4a. Multi-valued `param_file` truncates every path to one character

**Branch:** `ask/param-file-multi` (test-only, xfail) --
`tests/func/test_ControlFile.py::test_read_multiple_parameter_files`

PRMS supports multiple parameter files, and `control.xml`'s own
description reads "Pathname(s) for Parameter File(s)" -- but a
control listing two paths breaks ControlFile: `param_file` has
`numvals=1`, so its context derives as `scalar` (metadata.py), and
the scalar-context read allocates
`np.zeros(numval, dtype=np.str_)` -- dtype `'<U1'` -- truncating
every path to its first character before the values setter rejects
the array with TypeError. Verified workaround: flip the entry's
context to `array` before reading (single- and multi-valued files
then both round-trip).

## 4b. Partial parameter files parse as empty (silently)

**Branch:** `ask/partial-param-file` (test-only, xfail) --
`tests/func/test_ParameterFile.py::test_read_partial_parameter_file`

The secondary files of the multiple-parameter-file feature are
PARTIAL: parameter blocks only, no header/dimension sections (e.g. a
`transp_frost.param` carrying only spring_frost/fall_frost alongside
a full `myparam.param`). `ParameterFile._read`'s header scan
consumes the whole file looking for the `** Dimensions **` marker,
so every parameter block lands in `.headers` and the file yields
zero parameters with no warning. Verified workaround: synthesize the
missing sections (dimension sizes are known from the full file read
first) -- suggesting one fix shape: accept externally-supplied
dimensions when a file carries no Dimensions section. Related:
issue #39 (partial CONTROL file roundtrip).

## 5. `Control.dynamic_parameters` returns prose for some flags

**Branch:** `ask/dynamic-parameters-listing` (**fix included**,
suite green) --
`tests/func/test_Control.py::test_dynamic_parameters_are_parameter_names`

The listing's contract (pinned by the existing
`test_set_dynamic_parameter`) is parameter names, and 13 of the 15
dyn flags honor it -- but `dyn_fallfrost_flag=1` yields
`['file fallfrost_dynamic']` and `dyn_potet_flag` yields sentence
fragments (its valid_values meanings are prose with module lists,
comma-split). The branch fixes the `control.xml` valid_values data
and strips whitespace in `dyn_param_meaning`. Bonus observation left
unfixed (scope): `init_vars_from_file` also carries
`valid_value_type='parameter'` with prose meanings, so
`dynamic_parameters` reports the RESTART flag too (`=1` yields
`['yes']`).

## 6. Feature ask: a dynamic-parameter file reader

**No branch** -- there is nothing to xfail; this is a feature offer.

pyPRMS has no reader for PRMS dynamic-parameter files
(`dyn_ag_frac.param`, `spring_frost.dyn`, ...: free header, then
whitespace-separated rows `year month day v_0 ... v_{n-1}` at
irregular dates, each row holding until the next -- forward-fill
semantics). We carry a small pyPRMS-shaped implementation
(~90 lines, xarray out: `load_dynamic_parameter` + `forward_fill`,
pinned against pywatershed's reader) in
`pws_phoenix/incarnations/mpixarray/prms_translate/dyn_param.py` and
would gladly shape it into a PR wherever it fits (perhaps beside the
DataFile work on `DataFile_updates_pan`).

---

# Migration map (the pyPRMS split-out boundary)

The one-way rule stands: pws_phoenix core never imports
`prms_translate`, and the generic layer below never imports the
incarnation. What migrates INTO pyPRMS when the asks land, vs. what
is pws_phoenix-specific forever:

| prms_translate piece | Destination |
| --- | --- |
| `readers.py` f64 patch + metadata widen | dies when ask 1 lands |
| `readers.py` `_AG_PARAMETER_METADATA` / `_AG_CONTROL_METADATA` / actet injection | dies when ask 3 lands (the branch IS the migration) |
| `readers.py` `param_file` context flip | dies when ask 4a lands |
| `readers.py` partial-file header synthesis | dies when ask 4b lands |
| `readers.py` force_default strip | dies when ask 2 lands |
| `dyn_param.py` (reader + forward_fill) | migrates as the ask-6 PR |
| `cbh.py` object-dtype nhru coord drop | minor pyPRMS quirk; report alongside ask 3 or keep |
| `control.py` (class resolution, PrmsRunConfig) | **stays** (pws_phoenix semantics) |
| `parameters.py` (contract-driven packaging) | **stays** |
| `assemble.py` / `assemble_mpi.py` | **stays** |
| `preprocess.py` (stamp + verify) | **stays** |

Minor quirks not worth their own issues (mention opportunistically):
pyPRMS's Cbh stamps a 1-based object-dtype `nhru` index coordinate
(breaks zarr encoding downstream); `('one',)`-dim parameters return
python scalars where arrays would be uniform;
`.additional_modules` raises on NHM controls lacking the basinOut
flags.
