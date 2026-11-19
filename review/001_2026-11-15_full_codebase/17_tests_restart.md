<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Slice 17: the restart tests](#slice-17-the-restart-tests)
  - [Findings](#findings)
    - [17.1 `perfect_restart` compares the two runs once, after the last step: state that is lost at the restart and replaced before the end passes](#171-perfect_restart-compares-the-two-runs-once-after-the-last-step-state-that-is-lost-at-the-restart-and-replaced-before-the-end-passes)
    - [17.2 Nothing shows that the state at the restart point differs from a cold start's, so a missing flag is found only where the data happen to make it differ](#172-nothing-shows-that-the-state-at-the-restart-point-differs-from-a-cold-starts-so-a-missing-flag-is-found-only-where-the-data-happen-to-make-it-differ)
    - [17.3 The budget accumulators are restart variables and are never compared](#173-the-budget-accumulators-are-restart-variables-and-are-never-compared)
    - [17.4 `test_flags_are_variables_only` cannot fail, and both registry guards check whichever classes the session has created so far](#174-test_flags_are_variables_only-cannot-fail-and-both-registry-guards-check-whichever-classes-the-session-has-created-so-far)
    - [17.5 In `toy_run` a failure on rank 0 alone leaves the other ranks in `comm.Barrier()`; the module docstring says this cannot happen](#175-in-toy_run-a-failure-on-rank-0-alone-leaves-the-other-ranks-in-commbarrier-the-module-docstring-says-this-cannot-happen)
    - [17.6 No restart test reads what a warm-started run writes as output](#176-no-restart-test-reads-what-a-warm-started-run-writes-as-output)
    - [17.7 Under MPI the restart tests cover float64 state of the toys and of groundwater, written by MPI and read by both](#177-under-mpi-the-restart-tests-cover-float64-state-of-the-toys-and-of-groundwater-written-by-mpi-and-read-by-both)
    - [17.8 Guards of the restart and initial-state code with no test showing that they raise](#178-guards-of-the-restart-and-initial-state-code-with-no-test-showing-that-they-raise)
    - [17.9 Classes and configurations that no restart test runs](#179-classes-and-configurations-that-no-restart-test-runs)
    - [17.10 Supplied initial state is tested on the toys; on the ports only three variables are ever supplied, each whole and on its own](#1710-supplied-initial-state-is-tested-on-the-toys-on-the-ports-only-three-variables-are-ever-supplied-each-whole-and-on-its-own)
    - [17.11 Comments and docstrings that do not match the code, duplicated setup, style](#1711-comments-and-docstrings-that-do-not-match-the-code-duplicated-setup-style)
  - [Code findings of slices 1, 3, 5-13: why did no test catch this?](#code-findings-of-slices-1-3-5-13-why-did-no-test-catch-this)
  - [Checklist T, item by item](#checklist-t-item-by-item)
  - [Notes for later slices](#notes-for-later-slices)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Slice 17: the restart tests

- pws_phoenix commit: 8fed99b (branch `review_001`; read from
  `.git/HEAD` and `.git/refs/heads/review_001` as text). The earlier
  slices record that the commits since e33e217 are review commits.
  The 5 files total 1,784 lines, as in the plan.
- Paths (under `tests/`): `test_restart.py` (351),
  `test_restart_processes.py` (462),
  `test_restart_processes_ag.py` (260),
  `test_restart_flow_graph.py` (294), `test_restart_mpi.py` (417)
- Checklist: T. Covers code slices 1, 3, 5-13. Reviewed 2026-11-19.
- Method: `/code-review high` on the 5 files (10 candidates, and one
  it dropped as not confirmable: the comment on the silos, here
  17.11 item 2). Kept: all 10, as 17.3, 17.4 (the guard that cannot
  fail and the registry candidate, merged), 17.5, 17.6 (the zarr
  output of a warm start and the length-only check of the streamed
  output, merged), 17.7 (the hook state under MPI) and 17.11 (the
  test classes without `units`, the duplicated setup, the `sys.path`
  edits). Added by the read: 17.1, 17.2, 17.8, 17.9, 17.10. Then
  every file of the slice, whole, against checklist T.
  - Outside the slice, through CodeGraph and direct reads:
    `pws_phoenix/model.py:515-529`, `:542-645`, `:931-989`,
    `:993-1137`, `:1568-1603`, `:1640-1730`, `:1742-1875`;
    `pws_phoenix/process.py:314-363`, `:394-478`;
    `pws_phoenix/processes_concrete.py:103-180`;
    `pws_phoenix/hydrology/prms_stream_temp.py:478-576`,
    `:1132-1136`, `:1250-1300`, `:1379-1391`;
    `pws_phoenix/atmosphere/prms_atmosphere.py:786-857`;
    `pws_phoenix/hydrology/prms_soilzone_ag.py:1872-1879`;
    `pws_phoenix/hydrology/prms_runoff.py:1868-1880`;
    `tests/conftest.py:157-162`, `:179-186`, `:219-230`;
    `docs/PORTS.md:420-492`.
  - Before the review: `coverage_serial.txt` (whole); the headings
    of every finding of slices 1, 3, 5-13, 15, 16, and in full the
    findings and notes that name restart or supplied state (2.1,
    3.4, 3.5, 3.8, 3.9, 6.1, 15.17, and the notes for slice 17 at
    the end of the files of slices 3, 6-13 and 16).
- Run: the four serial files in env pwpx2, with
  `--error-for-skips`: 24 passed in 19 s. `test_restart_mpi.py` was
  not run. No test was changed to see whether it then fails, and no
  data file was read: a finding that depends on a parameter value
  says "not read" and names the run that would settle it (the last
  section).

Findings are ordered by consequence: first what the perfect-restart
recipe passes for a reason other than the one it names (17.1-17.4);
then a hang (17.5); then what the slice does not reach (17.6-17.10);
then comments and style (17.11).

## Findings

### 17.1 `perfect_restart` compares the two runs once, after the last step: state that is lost at the restart and replaced before the end passes

- Where: `tests/test_restart_processes.py:70-100`;
  `tests/test_restart.py:100-110`;
  `tests/test_restart_mpi.py:164-175`, `:377-381`
- Code:

      with make_model({"restart_read": restart_dir}) as m_bc:
          assert m_bc._start_index == idx_b
          m_bc.run(dt, np.int32(n_total - idx_b))

      for kk, proc_ac in m_ac.model_dict.items():
          proc_bc = m_bc.model_dict[kk]
          for nn in type(proc_ac).get_var_names():
              np.testing.assert_array_equal(
                  proc_bc[nn].values,
                  proc_ac[nn].values,

- Consequence: every restart test in the slice, serial and MPI,
  compares the state after `n_total - idx_b` resumed steps (30 in
  the full chain and the channel graph, 45 for groundwater, the Ag
  chain and STARFIT, 60 for stream temperature and ObsET, 90 for
  snow and atmosphere). The first resumed step, where a lost state
  shows, is never compared. Two cases:
  - Finding 6.1: a warm-started daily STARFIT node writes nan
    `lake_release` and `lake_spill` on its first resumed day and
    correct values from the next day on.
    `test_starfit_graph[starfit_daily]` passes (run above).
  - The stream-temperature silos. A silo is a circular buffer of
    `gw_tau` (or `ss_tau`) days of `seg_tave_air`, which is an
    input:

        gw_silo[jj, idx_gw] = seg_tave_air[jj]
        ...
        if idx_gw < tau_gw - 1:
            gw_index[jj] = idx_gw + 1
        else:
            gw_index[jj] = 0

    In the continuous run day `t` lands in position `t % tau`. In a
    warm start that restored no silo (all -99.9, position 0) it
    lands in `(t - idx_b) % tau`. Where `idx_b` is a multiple of
    `tau` and at least `tau` steps are resumed, the silo and its
    position are identical at the end with no restore at all.
    `test_stream_temp` has `idx_b` = 60 and 60 resumed steps;
    `test_full_chain` has 30 and 30. A `tau` of 30 meets both. The
    drb values of `gw_tau` and `ss_tau` were not read. On the days
    in between `seg_tave_gw` / `seg_tave_ss` differ; whether a trace
    reaches the last step through `seg_tave_water` was not followed.
  `Model.step(dt)` exists (`pws_phoenix/model.py:962`): in serial
  the continuous and the warm model can be stepped side by side and
  compared after every step. Under MPI there is no `step()`; the
  streamed output files of the two runs hold every step (17.6).
- Source: checklist T (a comparison of only some time steps); the
  note of slice 6
- Outcome: open

### 17.2 Nothing shows that the state at the restart point differs from a cold start's, so a missing flag is found only where the data happen to make it differ

- Where: `tests/test_restart_processes.py:72-73`, `:178-179`,
  `:214-215`, `:273-274`, `:457-462`;
  `tests/test_restart_processes_ag.py:224`, `:260`;
  `tests/test_restart_flow_graph.py:172`, `:294`
- Code:

      if idx_b is None:
          idx_b = n_total // 2

      # winter into spring: pack builds, knife edges exercised
      perfect_restart(make_model, np.float64(1.0), 180, tmp_path / "rst")

      # spans the istep0 transp initialization + season transitions
      perfect_restart(make_model, np.float64(1.0), 180, tmp_path / "rst")

- Consequence: the warm model runs `initialize()` before the restore
  (`pws_phoenix/model.py:136-138`), so a restart variable that has
  no flag starts the second half at its default. The comparison sees
  that only if the continuous run's value at `idx_b` differs from
  the default on some element, and the difference lasts to the end
  (17.1). No test reads a restart file's values (the toy test reads
  the names and `state_time`), and none compares the state at
  `idx_b` with a cold model's. What the comments say the window
  "exercises" is not asserted. So:
  - Whether a flag is complete is shown at one date of one domain.
    `docs/PORTS.md:438-440` records how thin that can be: `pk_temp`
    was "caught by the full-chain test at one knife-edge HRU".
  - `test_atmosphere` writes its restart after step 89 of 180, the
    full chain after step 29 of 60. Whether any HRU then has
    `transp_check == 1` and `tmax_sum > 0`, the state `transp_check`
    was flagged for (the note of slice 12), depends on `transp_beg`
    and the start date. Not read.
  - A flag that is NOT needed cannot be seen either: `ai` (11.4),
    `transp_on` in the frost leaves (12.6), `ag_soil_lower` (10.8)
    pass with or without the flag.
  - The flag sets of the nine NHM processes are pinned by no test.
    The exact sets are pinned for the Ag family and for the graph
    (`test_flags_derivation_pin`, twice). "Flags were cross-checked
    against pywatershed's own hand-maintained
    get_restart_variables() lists"
    (`tests/test_restart_processes.py:20-21`) was a check by hand.
- Source: checklist T (a comparison of only some elements and time
  steps); the notes of slices 11 and 12
- Outcome: open

### 17.3 The budget accumulators are restart variables and are never compared

- Where: `tests/test_restart_processes.py:85`;
  `tests/test_restart_mpi.py:167`, `:174`, `:380`;
  `pws_phoenix/process.py:425-426`, `:450-454`
- Code:

      for nn in type(proc_ac).get_var_names():

      def get_var_names(cls, include_budget: bool = False) -> tuple[str, ...]:

      for name, meta in cls.get_variables(include_budget=True).items()
      if meta.restart

- Consequence: `{Class}_{budget}_residual_accum` is `restart=True`,
  is in every restart file, and is updated on every step whatever
  `control["budget"]` says (`Model._check_budgets` returns early
  only for a process with no budget, `pws_phoenix/model.py:518`).
  `get_var_names()` without `include_budget=True` leaves it out of
  the comparison, in serial and under MPI (PRMSGroundwater declares
  a mass budget). If its restore broke (for example
  `initialize_budgets()` moved after `_read_restart`), the warm
  model would hold the sum of the second half only and every
  perfect-restart test would still pass. Only the name is pinned
  (`tests/test_restart_processes_ag.py:163`, `:177`, `:185`;
  `tests/test_restart_flow_graph.py:108`). The module docstring
  says "in EVERY variable of EVERY process".
- Source: /code-review, confirmed by reading (not run)
- Outcome: open

### 17.4 `test_flags_are_variables_only` cannot fail, and both registry guards check whichever classes the session has created so far

- Where: `tests/test_restart.py:324-351`;
  `pws_phoenix/process.py:321-323`, `:442-454`
- Code:

      for name, cls in Process._registry.items():
          for vv in cls.get_restart_variables():
              assert vv in cls.get_var_names(include_budget=True), f"{name}.{vv}"

      return tuple(
          name
          for name, meta in cls.get_variables(include_budget=True).items()
          if meta.restart
      )

- Consequence:
  - `get_restart_variables()` is a filter of
    `get_variables(include_budget=True)`, and `get_var_names(
    include_budget=True)` is the keys of the same dict. The assert
    is true for any class. The case the docstring names, a flag on a
    field that is not a variable (`restart=True` on an input or a
    parameter, finding 1.8), is dropped by the filter before the
    test looks. A guard that can fail walks every resolved field
    and asserts that one with `meta.restart` has kind `variable` or
    `budget_variable`.
  - `test_state_groups_are_restart_variables` imports three modules;
    `test_flags_are_variables_only` imports none. `pytest
    tests/test_restart.py` alone checks the toys, the classes of
    the file and those three modules. STARFIT's
    `required_initial_state` sits on node types, which are not
    `Process` classes; it reaches the registry only through a graph
    class that some test composed earlier in the session
    (`make_flow_graph` creates the class when called). The registry
    is keyed by class name (16.12), so of two classes with one name
    the guard sees the later.
- Source: /code-review (two candidates, merged); the note of
  slice 16
- Outcome: open

### 17.5 In `toy_run` a failure on rank 0 alone leaves the other ranks in `comm.Barrier()`; the module docstring says this cannot happen

- Where: `tests/test_restart_mpi.py:29-31`, `:196-240`
- Code:

      All collective MPI ops live in the module-scoped fixtures; test
      methods are pure (collective-free) asserts -- a failing rank can never
      interrupt a collective and hang the others.

      if comm.rank == 0:
          with xr.load_dataset(toy_paths["data_dir"] / "out_bc.nc") as ds:
      ...
      if comm.rank == 0:
          ...
          with Model(
              process_dict, {"restart_read": restart_dir}, maps=maps
          ) as m_serial:
              assert m_serial._start_index == TOY_N_AB
              m_serial.run(dt, np.int32(TOY_N_AC - TOY_N_AB))
      ...
      comm.Barrier()

- Consequence: the interop leg exists to show that a serial model
  reads the files an MPI run wrote. If it cannot (`_read_restart`
  raises, or the `assert` fails), the exception leaves the fixture
  on rank 0 only and ranks 1-3 wait in the `Barrier` at `:240`:
  `mpirun -n 4 pytest` hangs and reports nothing (15.13: no launch
  has a time limit). Earlier in the same fixture, `start_ok =
  m_bc._start_index == TOY_N_AB` shows the pattern that defers the
  assert to a test method; the rank-0 block does not follow it. A
  `try` around the block that stores the exception for a test
  method to raise would.
- Source: /code-review, confirmed by reading (not run)
- Outcome: open

### 17.6 No restart test reads what a warm-started run writes as output

- Where: `tests/test_restart_mpi.py:194-199`, `:260-265`; the four
  serial files (none names `output_serial_zarr`: text scan)
- Code:

      n_out_bc = None
      if comm.rank == 0:
          with xr.load_dataset(toy_paths["data_dir"] / "out_bc.nc") as ds:
              n_out_bc = int(ds.sizes["time"])

      assert toy_run["n_out_bc"] == TOY_N_AC - TOY_N_AB

- Consequence:
  - Finding 2.1 (`Output.collect_current_timestep` ignores its
    `time_index`, so a warm start writes its rows under the dates
    of the first rows) has no test: no serial restart test asks for
    output, and the MPI toy asks only for the distributed `flow`,
    so the rank-0 zarr `Output` of a warm-started `ModelMPI` is not
    built.
  - The streamed NetCDF of the warm run is checked for its length.
    Its `time` values and its `flow` rows are not compared with
    rows 30-59 of `out_ac.nc`, which the same fixture has written.
    A stream labelled from `time[0]`, or 30 rows of wrong values,
    passes `test_stream_starts_at_resume`.
- Source: /code-review (two candidates, merged)
- Outcome: open

### 17.7 Under MPI the restart tests cover float64 state of the toys and of groundwater, written by MPI and read by both

- Where: `tests/test_restart_mpi.py` (whole);
  `pws_phoenix/model.py:1791-1875`
- Code:

      state = proc.get_restart_state()
      if distributed and state:
          raise NotImplementedError(

      units = _resolved_fields(type(proc))[name].units
      ds_out[name] = (
          (grid,),
          np.concatenate(parts),
          {} if units is None else {"units": units},
      )

- Consequence: not reached by any MPI test:
  1. Restart state held in python attributes. On a replicated grid
     rank 0 writes `{process}__gw_silo` and every rank calls
     `set_restart_state`: `tests/test_prms_stream_temp_mpi.py` has
     no restart. The `NotImplementedError` for the distributed grid
     (`:1836-1842`) has no test.
  2. State that is not float64. Finding 3.9: `ModelMPI` declares
     every state variable float64 with a nan fill. Snow's bool
     flags (`iasw`, `lst`) and int64 flags (`iso`, `lso`, `int_alb`),
     and the atmosphere's int64 `transp_check`, never pass through
     an MPI write and a serial read, or back.
     This is why 3.9 was not caught.
  3. The other direction of the interop claim. A serial model reads
     MPI-written files (`test_serial_warm_start_from_mpi_files`).
     No `ModelMPI` reads serial-written files, which carry the grid
     dataset's coordinates and attrs (`ds_out[name] =
     disc.dataset[name]`). The docstring at `:1797-1798` says
     "serial and MPI runs can warm-start each other";
     `docs/PORTS.md:471-472` adds "(tested)".
  4. Units. The MPI write takes `units` from the declarations
     (quoted above). No test reads that attr, and no warm start
     under `control["units"]` uses an MPI-written file
     (`tests/test_units_check.py:329` is serial; its MPI file has
     no restart).
  5. Two processes on the distributed grid, and a restart name two
     processes share (`if name in ds_out: continue`): each MPI case
     has one process there.
  6. One rank count (15.14).
  Reached, though the file does not say how: the global time index
  of a warm start. No process in the file has a first-step block,
  but `Upper` reads `param_up_1[time.month - 1]` and the resumed
  toy window (steps 30-59 from Jan 1) runs into February, so an
  index counted from the start of the stream would change `flow`.
- Source: /code-review (item 1); checklist T (a serial test with
  no MPI counterpart where the MPI code differs); the notes of
  slices 3 and 16
- Outcome: open

### 17.8 Guards of the restart and initial-state code with no test showing that they raise

- Where: `pws_phoenix/model.py:1023-1024`, `:1049-1056`,
  `:1093-1116`, `:1584-1585`, `:1660`, `:1742-1750`, `:1761-1788`;
  `pws_phoenix/process.py:474`
- Code:

      if not state_times:
          raise ValueError(
              f"restart_read: no restart files found in {directory} "
      if len(state_times) != 1:
          raise ValueError(
              f"restart_read: inconsistent state times across grid "
      if where.size != 1:
          raise ValueError(
              f"restart state time {state_time} not found on this "
      if self._start_index >= self.ntime:
          raise ValueError(
              f"restart state time {state_time} is this model's "

- Consequence, serial (`coverage_serial.txt`: `1094`, `1099`,
  `1106`, `1113` never run):
  - The four raises quoted: no files and no restart variables;
    two grids with different state times; a state time that is not
    on the time axis; a state time on the last step.
  - `len(found) != 1` is reached with an empty directory only. Two
    files for one grid is what a second `write_restart` on a later
    date into the same directory leaves
    (`{date}_restart_{grid}.nc`); no test.
  - The variable-set check is tested with a variable removed, not
    with one added.
  - `1024` and `1050` (`continue` for a grid with no restart
    content) never run: every grid in every serial restart test has
    a restart variable.
  - Finding 3.4: no test calls `write_restart` before the first
    step. Finding 3.5: no test reads a restart under another
    `process_dict` key, or from a file without the `{process}__`
    variables. Finding 3.8: every supplied array and every restart
    variable in the tests has the grid's length.
  - `check_initial_state` is tested on supplied state. A restart
    file with a value the process refuses is not tested.
  - `pws_phoenix/process.py:474->exit`: the Model calls
    `set_restart_state` only with non-empty state
    (`pws_phoenix/model.py:1091`), so the base method's return
    without a raise cannot be reached from a Model.
- Consequence, MPI (no MPI file but `tests/test_units_check_mpi.py`
  names `pytest.raises`: text scan): the four raises of
  `_peek_restart_resume_index`; supplied state together with
  `restart_read`; a partial state group; missing required state;
  and `ModelMPI._raise_if_bad_initial_state`, the one collective
  raise among them (the others are the same on every rank by
  construction). The note of slice 16 asked whether
  `tests/test_restart_mpi.py` catches that raise by hand: it does
  not.
- Source: coverage, checklist T (a guard with no test)
- Outcome: open

### 17.9 Classes and configurations that no restart test runs

- Where: `tests/test_restart_processes.py:10-18`;
  `tests/test_restart_flow_graph.py:18-25`;
  `tests/test_restart_processes_ag.py:16-18`;
  `pws_phoenix/process.py:447-448`
- Code:

      - pass_through / obsin / source_sink: NO prognostic state (their
        accumulators are zeroed in prepare; obsin's within-day latch never
        crosses the day boundary); the combined starfit_source_sink adds
        none beyond the family storages.

      declarations ... The declaration-override seam applies: a variant's
      redeclaration replaces the base's flag.

- Consequence:
  - Process classes that are never warm-started and whose flag set
    is not pinned: `PRMSRunoffNoDprst`, `PRMSSoilzoneNoDprst`,
    `PRMSGroundwaterNoDprst`, `PRMSAtmosphereTranspFrost`,
    `PRMSAtmosphereTranspFrostDyn`, `PRMSStreamTempConstantShade`,
    `PRMSStreamTempSegHumidity`, `PRMSHydraulicGeometryFull`. A
    variant that redeclares a field without `restart=True` drops
    the base's flag without a message; only a pin or a restart run
    of that variant would show it. The frost leaves inherit the
    base's flag on `transp_on` (12.6) and are never restarted.
  - Node types never in a restart run: pass_through, obsin,
    source_sink, starfit_source_sink. "NO prognostic state" for
    them is the docstring's analysis, tested by nothing. No mixed
    graph is restarted (the restore writes whole `(nnodes,)` arrays,
    the nan rows of other types included), and STARFIT is restarted
    in a cms graph only (the note of slice 6; 6.5).
  - The coupled Ag pair is restarted with a static `ag_frac`; the
    dynamic `ag_frac` runs with ObsET alone. No restart crosses a
    change date in the coupled pair (10.1).
  - Every restart file in the slice is written by a cold-started
    model. No test writes a restart from a warm-started model and
    resumes from that, which is what a run in segments does.
- Source: checklist T; the notes of slices 6, 10 and 12
- Outcome: open

### 17.10 Supplied initial state is tested on the toys; on the ports only three variables are ever supplied, each whole and on its own

- Where: `tests/test_restart.py:150-321`;
  `tests/test_restart_processes.py:117-119`, `:430-436`;
  `tests/test_restart_flow_graph.py:275-292`;
  `tests/test_restart_mpi.py:136`, `:143-145`, `:365`
- Code:

      **cold_start_state(
          control, gwres_stor=gw_params["gwstor_init"]
      ),

      state_groups = (("soil_moist", "soil_rechr", "soil_lower"),)

- Consequence: the group, the required-state and the
  `check_initial_state` machinery are each tested with a class
  written in `tests/test_restart.py`. On the ports the tests supply
  `gwres_stor`, `seg_inflow` and `lake_storage_sub` (required, with
  its nan check). So:
  - No test supplies a declared state group of a port, whole or in
    part: the soil zone's three, runoff's five depression
    variables. The guard at `tests/test_restart.py:324-343` checks
    that the names are restart variables, not that supplying them
    works.
  - The findings that a group is missing or unchecked had no test
    to fail: canopy (7.4), `imperv_stor` / `hru_impervstor` (8.4),
    the snowpack (11.1), the Ag pair (10.8), an inconsistent
    `soil_lower` (9.2), `pref_flow_stor > pref_flow_max` (9.1),
    `seg_tave_water` (13.3: the stream-temperature tests name it
    only to choose a tolerance).
  - The three known gaps of `CLAUDE.md` ("three first-step blocks
    overwrite supplied state on a cold start"; 8.3 is a fourth) are
    pinned by no test, so a fix has nothing to turn green and the
    present behaviour can change unseen.
  - `test_supplied_state_overrides_initialize` asserts `level is
    not supplied.values`. It fails if the buffer were shared whole;
    `np.shares_memory` would also catch a view (16.10).
- Source: the notes of slices 7-11 and 13; checklist T
- Outcome: open

### 17.11 Comments and docstrings that do not match the code, duplicated setup, style

- Where: listed per item
- Code:

      # > 90 days so the gw/ss silos wrap real history through b
      perfect_restart(make_model, np.float64(1.0), 120, tmp_path / "rst")

      level = DataArrayMeta(
          kind="variable", dims=("space",), dtype=np.float64, restart=True
      )

      sys.path.insert(0, str(PYWS_ROOT))

- Consequence:
  1. `tests/test_restart_mpi.py:29-31` says a failing rank cannot
     hang the others; 17.5.
  2. `tests/test_restart_processes.py:273`: the run is 120 steps
     and `b` is step 60. Whether a silo has wrapped by step 60
     depends on `gw_tau` / `ss_tau` (not read), and 17.1 shows that
     for some values the restore is not tested at all.
  3. `tests/test_restart_processes.py:3-8`: "in EVERY variable of
     EVERY process" (17.3: not the budget accumulators), "A
     forgotten prognostic flag ... diverges here" (17.1, 17.2: only
     if the data make it).
  4. Float `DataArrayMeta` declarations without `units=`
     (`CLAUDE.md`, python convention 5): `tests/test_restart.py:153`,
     `:193`, `:196`, `:253`;
     `tests/test_restart_processes.py:285`. `tests/test_units.py`
     checks the package only.
  5. `HumidityCarrier` (`tests/test_restart_processes.py:282-296`)
     is a third copy in `tests/`; the package has the class
     (`pws_phoenix/prms_translate/assemble.py:71`, with
     `units="percent"`). The copy replaces it in the registry
     (16.12).
  6. Copies of code that exists elsewhere: the `vol_weights` loop
     (`tests/test_restart_processes.py:334-339`) is
     `pyws_domain(...)["weights"]` (`tests/conftest.py:181-186`);
     the STARFIT data preparation
     (`tests/test_restart_flow_graph.py:185-198`) is
     `tests/test_starfit_flow_node.py:97-99` and the lines around
     it; `_dynamic_ag_frac`
     (`tests/test_restart_processes_ag.py:128-151`) is
     `tests/test_prms_soilzone_ag_obs_et.py:170`. `perfect_restart`
     is imported from its file; these could be too.
  7. `_dynamic_ag_frac` puts the pywatershed clone at the front of
     `sys.path` on every call, for the rest of the session.
     `sys.path.append(str(pl.Path(__file__).parent))`
     (`tests/test_restart_processes_ag.py:28`,
     `tests/test_restart_flow_graph.py:36`) is not needed: the
     other three files import `conftest` without it.
  8. Four of the five files carry `skipif` marks on missing data
     (`tests/test_restart_processes.py:59`, and the `*_skipif`
     marks of the other three). Under `tests/run_tests.sh` a skip
     is an error; in the CI job without the data it is a skip
     (15.12).
  9. The tests assert the private `_start_index`; nothing public
     reports the step a warm start resumes at.
  10. No line is longer than the 79 of `ruff.toml` (counted with
      awk).
- Source: /code-review (items 4-7), checklist T
- Outcome: open

## Code findings of slices 1, 3, 5-13: why did no test catch this?

One line per finding that bears on restart or on initial state; the
other findings of these slices belong to slices 18-21. "No test"
means no test in the five files, by reading and the text scans
named above.

- 1.8: the guard written for it cannot fail (17.4).
- 2.1: no restart test asks for output (17.6).
- 3.4, 3.5, 3.8: no test of the case (17.8).
- 3.9: only float64 state crosses MPI and serial (17.7 item 2).
- 3.14: no test measures memory; the MPI cases have 60 and 765
  elements.
- 3.21: no restart test raises during a build and then looks at
  the open files.
- 6.1: the comparison is at the end of the run (17.1).
- 6.5, and the STARFIT types in a cfs graph: no restart test
  (17.9).
- 7.4, 8.3, 8.4, 9.1, 9.2, 10.8, 11.1, 13.3: no supplied state on
  those processes (17.10).
- 10.1: the coupled pair is restarted with a static `ag_frac`
  (17.9).
- 11.4, 12.6, and `ag_soil_lower` of 10.8: a flag that is not
  needed passes (17.2).
- 13.5: the silos are restored under the key that wrote them, in
  serial, and for some `tau` the restore is not tested (17.1,
  17.7 item 1, 17.8).

## Checklist T, item by item

- A raise or a branch that no test reaches: 17.8. Reached here and
  nowhere else in the serial tests: the restart write and read
  (`pws_phoenix/model.py:993-1137`, but for the lines of 17.8), the
  hooks of `PRMSStreamTemp` (`:1379-1391`), the three state guards
  (`:589-644`).
- A test that cannot fail: `test_flags_are_variables_only` (17.4).
  Passing for less than the docstring says: every perfect-restart
  test (17.1-17.3). The expected value of a perfect-restart test
  comes from the code under test by design: it shows that two runs
  agree, not that either is right, which is the port tests' job.
- A tolerance looser than the process's standard: none. Every
  comparison is `assert_array_equal`.
- A comparison where nan on both sides passes:
  `assert_array_equal` treats nan in the same place as equal, and
  both sides are runs of the model. A variable that is nan on both
  sides at the end (a row where a STARFIT field does not apply, a
  never-has-flow segment of `seg_tave_water`) is equal without
  being tested. That is correct for those rows; no test counts how
  much of a compared array is nan.
- A comparison of only some output variables, time steps or
  elements: the last step only (17.1); not the budget accumulators
  (17.3); the streamed output by its length (17.6).
- A serial test with no MPI counterpart where the MPI code differs:
  17.7. `tests/test_restart.py` has guard tests (flag change, empty
  directory, supplied state with a restart, state group, required
  state, `check_initial_state`); the MPI file has none (17.8).
- A guard with no test showing that it raises: 17.8.

## Notes for later slices

- Slice 18: `tests/test_restart_flow_graph.py` restarts a pure
  channel graph and isolated STARFIT outlets. Does any FlowGraph
  test show that obsin and source_sink carry nothing across a day
  (17.9)? `test_starfit_graph` is the one place the nan check of
  `lake_storage_sub` is shown to raise.
- Slices 19, 20: no test in this slice supplies a state group of a
  port (17.10); do the port tests? Do any pin the flag set of a
  class (17.2, 17.9)?
- Slice 21: do the stream-temperature tests compare `seg_tave_gw` /
  `seg_tave_ss` on the first `tau` days, when the silos are part
  full (17.1)?
- Fix phase, runs that settle what this slice did not read:
  - `gw_tau`, `ss_tau` of `drb_2yr/parameters_PRMSStreamTemp.nc`
    (17.1; 17.11 item 2): is 60, or 30, a multiple of either?
  - In `perfect_restart`, compare each restart variable in the file
    written at `idx_b` with a cold model's value after
    `initialize()`, and list the variables that are equal
    everywhere (17.2). For `test_atmosphere`: the count of HRUs
    with `transp_check == 1` at `idx_b`.
  - In `perfect_restart`, step the two models side by side and
    compare after each step (17.1); expect
    `test_starfit_graph[starfit_daily]` to fail on 6.1.
  - Add `include_budget=True` at
    `tests/test_restart_processes.py:85` and run (17.3).
  - Remove `set_restart_state` from the read
    (`pws_phoenix/model.py:1091-1092`) and run `test_stream_temp`
    and `test_full_chain` (17.1).
