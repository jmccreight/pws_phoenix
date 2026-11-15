<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Slice 1: foundations](#slice-1-foundations)
  - [Findings](#findings)
    - [1.1 A `units_patch` entry is applied again to an array that was already converted in place](#11-a-units_patch-entry-is-applied-again-to-an-array-that-was-already-converted-in-place)
    - [1.2 An absolute temperature converts to a temperature difference, and back, without an error](#12-an-absolute-temperature-converts-to-a-temperature-difference-and-back-without-an-error)
    - [1.3 `same` treats `radians` as equal to `fraction` and `none`; `degrees` converts to them](#13-same-treats-radians-as-equal-to-fraction-and-none-degrees-converts-to-them)
    - [1.4 `Time` accepts a time axis that is not datetime64, and an index outside the axis](#14-time-accepts-a-time-axis-that-is-not-datetime64-and-an-index-outside-the-axis)
    - [1.5 A required budget component declared as an empty tuple passes validation and fails at the first step](#15-a-required-budget-component-declared-as-an-empty-tuple-passes-validation-and-fails-at-the-first-step)
    - [1.6 A budget term's kind, dims and dtype are not validated, though the kernel needs identical 1-D arrays](#16-a-budget-terms-kind-dims-and-dtype-are-not-validated-though-the-kernel-needs-identical-1-d-arrays)
    - [1.7 The budget warning reports positions within one rank's block under MPI, without the rank](#17-the-budget-warning-reports-positions-within-one-ranks-block-under-mpi-without-the-rank)
    - [1.8 `DataArrayMeta` validates nothing, and its `units` comment contradicts the closed vocabulary](#18-dataarraymeta-validates-nothing-and-its-units-comment-contradicts-the-closed-vocabulary)
    - [1.9 Two Process classes with the same name replace each other in the registry without a message](#19-two-process-classes-with-the-same-name-replace-each-other-in-the-registry-without-a-message)
    - [1.10 The toy kernels allocate one array per step; the docstrings say they allocate none](#110-the-toy-kernels-allocate-one-array-per-step-the-docstrings-say-they-allocate-none)
    - [1.11 `_resolved_fields` recomputes on every call, and re-validates the budgets each time](#111-_resolved_fields-recomputes-on-every-call-and-re-validates-the-budgets-each-time)
    - [1.12 The budget kernels are compiled on every run](#112-the-budget-kernels-are-compiled-on-every-run)
    - [1.13 The reason given for `autoconvert_offset_to_baseunit=True` is not what the option does](#113-the-reason-given-for-autoconvert_offset_to_baseunittrue-is-not-what-the-option-does)
    - [1.14 The module docstring of `clock.py` names another file](#114-the-module-docstring-of-clockpy-names-another-file)
  - [Checklist A, item by item](#checklist-a-item-by-item)
  - [Notes for later slices](#notes-for-later-slices)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Slice 1: foundations

- pws_phoenix commit: 50fc6cf (branch `review_001`; the package is
  unchanged from e33e217).
- Paths: `pws_phoenix/{clock,process,units,budget,processes_concrete}.py`
- Checklist: A. Reviewed 2026-11-15.
- Method: `/code-review high` on the five files, then a read of each
  file against checklist A. The session executed nothing; JLM ran
  two one-line checks of `units.py` afterwards (recorded in 1.2 and
  1.3). A finding marked "not run" rests on reading how pint or
  numba behaves and needs a short run to confirm before it is fixed.

Findings are ordered by consequence, worst first.

## Findings

### 1.1 A `units_patch` entry is applied again to an array that was already converted in place

- Where: `pws_phoenix/units.py:250` (and `:217`, `:227`)
- Code:

      supplied = patch.get(name, attrs.get("units"))
      ...
      apply_affine(da.values, scale, offset, where=where)
      da.attrs["units"] = declared

- Consequence: with `control = {"units": "convert", "units_patch":
  {"hru_area": "feet"}}` and an in-memory parameter dataset, the
  first build converts the caller's array (it is the model's buffer)
  and sets its attrs to the declared units. A second build from the
  same dataset and control (a rebuild in a notebook, a serial run
  then an MPI run, a calibration loop) reads `patch["hru_area"]`
  first, ignores the corrected attrs, and multiplies again. No error:
  `apply_affine` re-enables writing on the read-only array. The same
  precedence applies to a restart file read through `conversion`
  (`pws_phoenix/model.py:1073`): a patch written for a parameter of
  the same name rescales correctly labelled restart state.
- Source: /code-review
- Outcome: open

### 1.2 An absolute temperature converts to a temperature difference, and back, without an error

- Where: `pws_phoenix/units.py:270`
- Code:

      return affine(supplied, declared)  # raises if incompatible

- Consequence: claimed that pint gives `degC` and `delta_degC` the
  same dimensionality, so `affine("degC", "delta_degC")` returns
  `(1.0, 273.15)` and `"convert"` adds 273.15 in place.
- Run by JLM, 2026-11-15 (env pwpx2): the claim is FALSE.
  `units.affine('degC', 'delta_degC')` raises `ValueError: units
  'degC' cannot be converted to 'delta_degC': Cannot convert from
  'degree_Celsius' to 'delta_degree_Celsius'`. The reverse,
  `units.affine('delta_degC', 'degC')`, raises the same way.
- Source: /code-review
- Outcome: withdrawn (not a defect)

### 1.3 `same` treats `radians` as equal to `fraction` and `none`; `degrees` converts to them

- Where: `pws_phoenix/units.py:188-192`
- Code:

      if qd.dimensionality != qs.dimensionality:
          return False
      scale, offset = affine(supplied, declared)
      return math.isclose(scale, 1.0, rel_tol=1e-12) and abs(offset) < 1e-12

- Consequence: pint defines the radian as dimensionless with scale 1.
  Under `"check"`, a field declared `units="radians"`
  (`pws_phoenix/hydrology/prms_stream_temp.py:1404`) supplied as
  `none` or `fraction` passes, and the reverse passes. Under
  `"convert"`, an array labelled `degrees` supplied for a `fraction`
  or `none` field is multiplied by pi/180 in place instead of
  raising.
- Run by JLM, 2026-11-15 (env pwpx2): CONFIRMED.
  `units.same('radians', 'none')` and
  `units.same('radians', 'fraction')` return `True`;
  `units.affine('degrees', 'fraction')` returns
  `(0.017453292519943295, 0.0)`.
- Source: /code-review
- Outcome: open

### 1.4 `Time` accepts a time axis that is not datetime64, and an index outside the axis

- Where: `pws_phoenix/clock.py:38-43`, `:101-102`
- Code:

      self._times: np.ndarray = np.asarray(arr)
      ...
      self.current_index = index
      date = self.current.astype("datetime64[s]").astype(datetime.datetime)

- Consequence: an integer or float time coordinate (a file opened
  with `decode_times=False`, a `time` variable without a `units`
  attribute, a toy input with `time = np.arange(n)`) is cast as
  seconds since 1970-01-01. Every step then reports year 1970,
  month 1, day of year 1, so `Upper.calculate` reads `param_up_1[0]`
  on every step, with no error. A cftime axis fails with an
  `AttributeError` that does not name the time axis. `set_index(-1)`
  silently selects the last time; an index past the end raises
  `IndexError` after `current_index` was already changed. Nothing
  checks that the axis is daily or increasing, which the class
  docstring states.
- Source: /code-review
- Outcome: open

### 1.5 A required budget component declared as an empty tuple passes validation and fails at the first step

- Where: `pws_phoenix/process.py:170`, `pws_phoenix/budget.py:114-118`,
  `:156`
- Code:

      missing = set(budget.REQUIRED_COMPONENTS) - set(terms)
      ...
      self.sum_names = {
          cc: field_name(class_name, budget_name, f"{cc}_sum")
          for cc in COMPONENTS
          if self.terms[cc]
      }
      ...
      inputs_sum = refs[self.sum_names["inputs"]]

- Consequence: `budgets = {"mass": {"inputs": (), "outputs": ("q",),
  "storage_changes": ("dS",)}}` (a store that only drains) passes
  the key check. No `inputs_sum` field is generated, and the first
  step raises `KeyError: 'inputs'` from `Budget.calculate`. This
  happens with `control["budget"] = None` too:
  `Model._check_budgets` (`pws_phoenix/model.py:515`) calls
  `check_budgets` for every process that declares a budget.
- Source: /code-review
- Outcome: open

### 1.6 A budget term's kind, dims and dtype are not validated, though the kernel needs identical 1-D arrays

- Where: `pws_phoenix/process.py:178-185`, `pws_phoenix/budget.py:56-63`
- Code:

      for name in names:
          if name not in fields:
              raise ValueError(...)
          units.add(fields[name].units)
      ...
      for tt in terms:
          acc += tt[ii]

- Consequence: numba types a read-only array differently from a
  writable one, and `for tt in terms` cannot loop over a tuple of
  mixed types. Example: `PRMSGroundwater` with
  `"inputs": ("soil_to_gw", "ssr_to_gw", "dprst_seep_hru")`, run with
  `PRMSRunoff` in the model (so `dprst_seep_hru` is a writable
  variable) while `soil_to_gw` and `ssr_to_gw` come from files
  (read-only `Input` buffers). The component mixes read-only and
  writable arrays, and the first step raises a numba `TypingError`
  rather than an error at declaration. A term that is a `parameter`
  (frozen), a float32 field, or a `("month", "space")` field fails
  the same way. The generated fields are fixed at `dims=("space",)`,
  float64. Not run.
- Source: /code-review
- Outcome: open

### 1.7 The budget warning reports positions within one rank's block under MPI, without the rank

- Where: `pws_phoenix/budget.py:178-182`
- Code:

      where = np.flatnonzero(~close)
      warnings.warn(
          f"budget '{self.residual_name}' imbalanced at {time.current} "
          f"at {n_bad} location(s), max |residual| = {max_abs:.3e}, "
          f"first indices {where[:10].tolist()}",

- Consequence: in a serial run the indices are positions on the
  grid. Under MPI each rank warns for its own block
  (`pws_phoenix/model.py:1721`), so `first indices [3, 7]` from rank
  2 means positions 3 and 7 of rank 2's block. The message names
  neither the rank nor the block's offset, so the locations cannot
  be found from it, and `n_bad` and `max |residual|` are per rank.
  The comment at `pws_phoenix/model.py:520` relies on this warning to
  carry the indices that the raise does not.
- Source: checklist A (a serial / MPI difference beyond the table)
- Outcome: open

### 1.8 `DataArrayMeta` validates nothing, and its `units` comment contradicts the closed vocabulary

- Where: `pws_phoenix/process.py:114-118`
- Code:

      # Physical units, free text (e.g. "inches", "cubic feet"). Optional
      # in general; REQUIRED on every term of a declared budget, and all
      ...
      units: str | None = None

- Consequence: the rule (every float field declares a vocabulary key)
  is enforced only by `tests/test_units.py`, so a Process defined
  outside the repository is not held to it. With `units="mm"` and
  `control["units"] = "check"`: if the supplied array also says
  `mm`, `conversion` blames the supplied array ("not in the
  pws_phoenix vocabulary"); for any other supplied key, `same`
  raises a bare `KeyError: 'mm'` at `pws_phoenix/units.py:186`. Also
  unchecked at declaration: `restart=True` on a `mutable_input` or
  `parameter_internal` is ignored by `get_restart_variables` without
  a message; a misspelled `kind` puts the field in no category; a
  `dtype` outside `_FILL_VALUE` (`np.float32`, `float`) raises a
  bare `KeyError` at `pws_phoenix/model.py:794`.
- Source: /code-review
- Outcome: open

### 1.9 Two Process classes with the same name replace each other in the registry without a message

- Where: `pws_phoenix/process.py:323`
- Code:

      Process._registry[cls.__name__] = cls

- Consequence: the class imported last wins, and
  `config.load_model_yaml` (`pws_phoenix/config.py:147`) resolves a
  yaml class name to it. `make_flow_graph` documents this for
  composed classes (`pws_phoenix/flow_graph.py:374-376`); for an
  ordinary subclass nothing says it. Generated budget fields are
  prefixed with `cls.__name__` (`budget.field_name`), so two
  same-named classes on one grid would also share their budget
  variables. Redefining a class in a notebook cell is a legitimate
  overwrite, so a fix needs to tell the two cases apart (for example
  by module).
- Source: checklist A (an error that passes silently)
- Outcome: open

### 1.10 The toy kernels allocate one array per step; the docstrings say they allocate none

- Where: `pws_phoenix/processes_concrete.py:89`, `:167-171`;
  docstrings at `pws_phoenix/processes_concrete.py:14-15` and
  `pws_phoenix/process.py:32-33`, `:271-272`
- Code:

      flow[:] = flow_previous * np.float64(0.95) + forcing_up * param_up_1

      IN PLACE; _calculate returns nothing and allocates nothing per step
      (numba fuses the array expressions, eliminating the temporaries).

- Consequence: numba fuses the right-hand side into one loop, but
  that loop still allocates its result, which is then copied into
  `flow[:]`. Each step allocates one `(space,)` array per kernel.
  This is the template in the `Process` docstring, so a kernel
  written from it carries an allocation that prime directive 2
  forbids. The ports use an element loop with scalar temporaries and
  have none. Not measured.
- Source: /code-review
- Outcome: open

### 1.11 `_resolved_fields` recomputes on every call, and re-validates the budgets each time

- Where: `pws_phoenix/process.py:138-154`
- Code:

      for cc in _proc_subclass_mro(cls):
          for name, val in vars(cc).items():
              if isinstance(val, DataArrayMeta):
                  fields[name] = val
      fields.update(_budget_fields(cls, fields))

- Consequence: every `get_*` classmethod and every
  `_resolved_fields(cls)[name].units` walks the class hierarchy,
  rebuilds a `budget.Budget` and returns new generated
  `DataArrayMeta` objects. `pws_phoenix/model.py:724` and `:750` call
  it once per parameter and per input, so build work grows with the
  square of the field count. Build time only, no effect on results.
  The comment at `pws_phoenix/process.py:116-117` says the budget
  check runs "when the class's fields are first resolved"; it runs
  on every resolve. A cache would also make a class edited after its
  first use (a test that sets `budgets` on a class) keep the old
  fields, which a fix must handle.
- Source: /code-review
- Outcome: open

### 1.12 The budget kernels are compiled on every run

- Where: `pws_phoenix/budget.py:56`, `:66`
- Code:

      @numba.njit
      def _sum_terms(out: np.ndarray, terms: tuple) -> None:

- Consequence: the process kernels use
  `@numba.njit(cache=NUMBA_CACHE)`; these two do not. Each run
  compiles `_balance` once and `_sum_terms` once per distinct number
  of terms, on every rank under MPI. Time only.
- Source: /code-review
- Outcome: open

### 1.13 The reason given for `autoconvert_offset_to_baseunit=True` is not what the option does

- Where: `pws_phoenix/units.py:129-136`
- Code:

      and only here). ``autoconvert_offset_to_baseunit`` lets absolute
      temperatures convert (degF -> degC) without an explicit delta."""
      ...
      _registry = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)

- Consequence: pint converts `degF` to `degC` with `.to()` without
  this option. What the option changes is arithmetic: multiplying or
  dividing an absolute temperature converts it to kelvin first
  instead of raising pint's `OffsetUnitCalculusError`. The registry
  is shared with the kernel units tests, so there a product with an
  absolute temperature no longer raises; whether the bit-for-bit
  comparison of magnitudes catches every such case is a question for
  slice 15. Not run.
- Source: checklist A (a check that cannot fail)
- Outcome: open

### 1.14 The module docstring of `clock.py` names another file

- Where: `pws_phoenix/clock.py:1-5`
- Code:

      """
      globals.py
      ==========
      Global runtime objects shared across a model run. Deliberately split (see
      pws_phoenix/CLAUDE.md):

- Consequence: the file is `clock.py`. The docstring of `process.py`
  (`:23-24`) likewise lists the kinds as "parameter / input /
  mutable_input / variable", without `parameter_internal` and
  `budget_variable`. Documentation only.
- Source: checklist A
- Outcome: open

## Checklist A, item by item

- Hidden copy: none. `np.asarray(arr)` (`pws_phoenix/clock.py:40`)
  does not copy an ndarray; `self._refs["param_up_1"][time.month - 1]`
  is a view.
- Allocation outside the three allowed classes: 1.10. The failure
  path of `Budget.calculate` allocates and says so in a comment.
  `Budget.calculate` builds one tuple of array references per
  component per step (`tuple(refs[tt] for tt in self.terms[cc])`):
  references, not arrays.
- `self._obj[name].values` inside a step method: none. The only
  lookup is the capture in `Process.__init__`.
- A raise on one rank alone, or a collective inside a rank-0 branch:
  none in these files. `check_budgets` returns a count and leaves
  the raise to the Model.
- Serial / MPI difference beyond the table: 1.7.
- An error that passes silently, or a check that cannot fail: 1.1,
  1.3, 1.4, 1.8, 1.9, 1.13.

## Notes for later slices

- Slice 3 (`pws_phoenix/model.py`): under MPI,
  `ModelMPI._raise_if_imbalanced` reduces the count with
  `op=MPI.MAX`, and the message then reads "{n_bad} location(s)":
  the largest count on one rank, not the total.
- Slice 3: `convertible=inp.data.variable._in_memory`
  (`pws_phoenix/model.py:751`) reads a private xarray attribute.
  `units.reconcile` converts `da.values`; for an array that is not in
  memory that would convert a temporary and still set the attrs, so
  the guard depends entirely on this attribute.
- Slice 5: whether a `io ...` placeholder can survive on a composed
  class (`same` returns True for two equal placeholder strings).
- Slice 15: finding 1.13, and why no test caught 1.1, 1.3, 1.5.
