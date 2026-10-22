# Maintenance ledger

Things to watch for and potentially fix later: known hazards kept
verbatim for parity with pywatershed/PRMS, each with the condition
that makes it bite and the detector that would catch it. One entry
per item; delete the entry when the item is resolved.

## PRMSSnow: `calc_snowevap` hands mass to `pkwater_equiv` but not to `pk_ice` / `freeh2o`

`pws_phoenix/hydrology/prms_snow.py`, end of `calc_snowevap` (PRMS
`snowcomp.f90` verbatim, via pywatershed):

```python
    avail_et = potet - hru_intcpevap - snow_evap
    if avail_et < 0.0:
        snow_evap = snow_evap + avail_et
        pkwater_equiv = pkwater_equiv - avail_et
```

(and the `snow_evap < 0.0` correction just above it). The kernel
tracks `pkwater_equiv` SEPARATELY from `pk_ice + freeh2o`; these
corrections move mass into only one of the two, so the mass budget
(storage terms = `pk_ice_change`, `freeh2o_change`) opens by
`|avail_et|` on any HRU-day that reaches the branch.

- **Reachable only if** `potet_sublim * snowcov_area > 1` or
  `hru_intcpevap > potet`: with `ez = potet_sublim * potet *
  snowcov_area - hru_intcpevap`, `avail_et >= 0` in every branch
  otherwise. PRMS's `potet_sublim` range is below 1 and PRMSCanopy
  bounds `hru_intcpevap` by `potet` (epan_coef hardwired 1.0), so it
  is dead on nhm domains.
- **Detector:** `control["budget"] = "error"` (date, max |residual|,
  indices) -- the PRMSSnow mass budget is declared (Oct 2026).
- **Not fixed** because PRMS and pywatershed carry the same lines and
  no data reach it; changing `pk_ice`/`freeh2o` here is a physics
  decision that would diverge from both on any domain that does. The
  broader irritant -- derived `pkwater_equiv` tracked beside the
  storages it should equal -- is upstream's design. The same
  tracker split DID bite once, in `calc_ppt_to_pack` 1.1.3 (`freeh2o
  = net_rain - pndz`); that one was reachable on drb and is fixed
  (PORTS.md "The snow story").
