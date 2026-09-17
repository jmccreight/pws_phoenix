# Cascades: how they work, and alternatives

Notes from a discussion with Claude, 2026-09-09, on branch
cascades_research. Recorded in memory (cascades-research).

## 1. How cascades work (pywatershed / PRMS)

### Topology is static and precomputed

`pywatershed/utils/preprocess_cascades.py` builds a directed acyclic
graph from the `cascade` parameters. Each HRU has `ncascade_hru` links,
each to a downslope HRU (`hru_down > 0`) or a stream segment
(`hru_down < 0`), carrying a fixed area fraction `hru_down_frac` (share
of the upslope HRU's outflow) and `hru_down_fracwt` (the same share
converted to a depth over the downslope HRU's area). `hru_route_order`
is a topological sort, upslope before downslope. Nothing about slope,
distance, roughness, or wetness enters the routing; the fractions never
change.

### Within one step, water crosses the whole network instantly

The kernel loops over HRUs in route order
(`pywatershed/hydrology/prms_runoff.py:783`). Each HRU computes its
local runoff using its own forcing plus whatever upslope HRUs deposited
*this step* in `upslope_hortonian`, then pushes its outflow to its
receivers before they are computed (`_run_cascade_sroff`,
`pywatershed/hydrology/prms_runoff.py:1528`).

So travel time is not one element per timestep; it is zero across the
entire chain. A cascade of 300 grid cells delivers to the stream on the
same day as a cascade of 1.

### All delay comes from re-storage, not transport

The receiving process decides what happens to cascaded water:

- Hortonian runoff arrives as extra water available at the surface of
  the downslope HRU: it can infiltrate, be partitioned by the
  contributing-area function, or run off again.
- Interflow and Dunnian runoff arrive as inflow to the downslope HRU's
  capillary reservoir (`pywatershed/hydrology/prms_soilzone.py:988`), so
  they must refill soil, spill to the gravity reservoir, and exit
  through the soilzone's own coefficients before moving on.
- Groundwater (`cascadegw_flag`, `gw_upslope`) arrives as inflow to the
  downslope groundwater reservoir. This third cascade is not ported in
  pywatershed; only the metadata exists.

The effective lateral travel time is therefore the sum of storage
residence times along the path, parameterized by soil and groundwater
coefficients rather than by geometry.

### Consequences

- No state in transit: no storage on the hillslope, no backwater, no
  depth. A wet flat receiver cannot slow anything.
- The serial route-order loop cannot be parallelized
  (`_nb_parallel_ok = False`), which is what makes gridded domains slow.
- On a grid with daily steps and cells under a kilometer, instant
  traversal is defensible for overland flow (which does travel
  kilometers per day) and poor for interflow and groundwater, where the
  per-cell re-storage becomes the only physics.

## 2. Could 2D diffusive wave replace cascades?

Short answer: a gridded diffusive wave can replace the two surface
flows (Hortonian, Dunnian), not interflow or groundwater. Those two
need a saturated-subsurface lateral solver. Both fit the same
explicit-grid engine, so the practical design is one engine with two
conductance laws.

### What diffusive wave gives you

A depth state per cell, velocity from slope and Manning roughness, and
mass in transit. Travel time becomes a function of geometry and depth
instead of storage coefficients. The cost is sub-stepping: the stable
step scales with cell size (WRF-Hydro uses about 10 seconds at 250 m),
so a daily model step becomes thousands of router sub-steps. Cost is
roughly cells times sub-steps, which grows like the inverse cube of
cell size. The square-grid restriction is what makes it a vectorized
stencil, hence Landlab's speed.

Landlab's `OverlandFlow` is the de Almeida local-inertial scheme, not
pure diffusive wave; same role, slightly looser stability limit. It
requires a raster grid.

### Why not interflow and groundwater

Diffusive wave has no porosity and a velocity scale of meters per
second; interflow moves meters per day through a saturated layer. The
right equation is Boussinesq/Dupuit: a nonlinear diffusion of saturated
thickness with hydraulic conductivity as the conductance. Landlab has
this too (`GroundwaterDupuitPercolator`, explicit, any grid), and
WRF-Hydro's subsurface lateral flow is the same idea (DHSVM-style). For
groundwater proper, MODFLOW 6 is the obvious solver and pywatershed is
already headed there via the GSFLOW work.

### Coupling to a daily model

One-way per day is workable: HRU runoff depth is a uniform source over
the day, the router sub-cycles, stream inflow per segment and
end-of-day ponded depth come back, and ponded depth re-infiltrates at
the next day's soil step. Two-way feedback within the day
(re-infiltration during routing) needs the soil process at the router's
step, which PRMS's structure does not offer.

### HRU to grid and back

Conservative for the surface case by construction: an intersection
weight matrix spreads HRU depth uniformly over its cells and aggregates
ponded depth and re-infiltration back by area. Cells must be finer than
HRUs; otherwise a cell spans several HRUs and the routing scale is
coarser than the runoff-generation scale, so the DEM slope inside the
cell means nothing.

The subsurface case does not map cleanly: a Dupuit solver's state is a
water table per cell, and the HRU soilzone's state is bulk storage.
Mapping between them is nonlinear, and the soilzone would have to give
up its lateral terms and accept exfiltration as an aggregated input.
For gridded domains none of this arises, which argues for making the
HRU path a legacy shim rather than the design center.

### Caveats that change the decision

- Daily-mean rain intensity undercuts overland-flow physics; depths
  from a daily rate are tiny, and infiltration excess at daily steps is
  already a calibration artifact. WRF-Hydro runs at hourly or finer
  forcing.
- Hillslope Manning n and effective conductivity are as poorly
  constrained as cascade storage coefficients, so this trades a
  calibrated lumped delay for a calibrated physical one. Under the
  port-fidelity rule this is a new process, and the cascade classes
  stay for PRMS comparison.

## 3. Why SUMMA has (almost) no lateral flow

SUMMA has less lateral flow than PRMS, but not none.

### What it does have

Within a GRU (grouped response unit, its unit of computation), HRUs can
be chained hillslope-style through a `downHRUindex` attribute. Only the
per-layer saturated-zone outflow (`mLayerColumnOutflow`) of an upslope
HRU becomes inflow to the downslope HRU within the same step; surface
runoff never crosses to a neighbor (code in section 5). It is the same
instantaneous idea as PRMS, with a single receiver and no area
fractions. It is rarely used; most setups make each GRU one HRU.

At the GRU scale, lateral drainage is a parameterization, not a
transport: the `qbaseTopmodel` baseflow option (power-law
transmissivity times slope times hillslope width; SUMMA uses no
topographic index anywhere, and its saturated area is a root-zone
moisture function) or a simple bucket. Runoff leaving a GRU goes to
mizuRoute, which routes only streamflow. Hillslope water never crosses
a GRU boundary.

### Why it was designed that way

SUMMA is a vertical-column model whose point is to swap process
alternatives (snow, soil, canopy, Richards versus bucket) inside a
single consistent numerical framework: one nonlinear state vector per
GRU, solved implicitly. A lateral term between columns couples every
GRU into one Jacobian, which destroys the independence that lets GRUs
run in parallel and lets the solver be a per-column Newton iteration.

Clark's stated position, from the CONUS-scale and TOPMODEL-lineage
work, is that at the scales SUMMA targets, subsurface lateral
redistribution is better handled as a subgrid parameterization of
saturated area than as resolved flow between units, and that channel
routing is a separate model (mizuRoute) rather than a process in the
column.

### The practical reading

SUMMA answers "which vertical process representation matters" and
leaves "where does the water go sideways" to attributes (the GRU
topographic index) and to routing. For a fine grid where a cell is
smaller than a hillslope, that assumption fails in the same way PRMS
cascades on a grid fail, just silently: each column drains straight
down to its own stream link. Nobody has bolted a Dupuit or
diffusive-wave layer onto SUMMA, because of the coupled-solver cost
above. If pywatershed goes that way it will be ahead of SUMMA on this
axis, not copying it.

Checked against the SUMMA source on GitHub master, 2026-09-10 (file
links in section 5).

## 4. Empirical vertical vs lateral partitioning, and its controls

From memory, not freshly checked against the papers; verify before
citing. The literature answers at two scales, and the answers differ.

### Catchment to global scale

Here "vertical vs lateral" is the evapotranspiration vs runoff
partition, since all runoff traveled laterally to reach a channel.
Rough global land budget as a fraction of precipitation:

| flux                                                | fraction of P | source                                        |
| --------------------------------------------------- | ------------- | --------------------------------------------- |
| evapotranspiration (vertical, up)                   | 0.60 to 0.65  | Oki and Kanae 2006; Rodell et al. 2015        |
| runoff (lateral)                                    | 0.35 to 0.40  | same                                          |
| groundwater recharge (vertical down, later lateral) | 0.10 to 0.15  | Döll and Fiedler 2008; Moeck et al. 2020      |
| baseflow share of runoff                            | about 0.5     | Beck et al. 2013 global baseflow index        |

Isotope hydrograph separation finds storm peaks are mostly "old"
pre-event water (Klaus and McDonnell 2013 review). So even the lateral
fraction is mainly subsurface displacement, not overland flow, outside
drylands and compacted or urban surfaces.

Main control: the Budyko framework. The aridity index (potential ET
over P) explains most between-catchment variance in runoff ratio.
Second-order controls from the Budyko-deviation literature: snow
fraction (Berghuijs et al. 2014), precipitation seasonality and
intensity, root-zone storage capacity and vegetation (Zhang et al.
2001), slope and relief.

### Hillslope and plot scale

This is the scale relevant to cascades. Infiltrated water moves
vertically until something impedes it. Lateral subsurface flow needs
three things together: a permeability contrast (bedrock, fragipan,
argillic horizon, frozen ground), slope, and a wetness threshold. The
"fill and spill" result at Panola is the canonical empirical case:
Tromp-van Meerveld and McDonnell 2006 found lateral subsurface
stormflow negligible below a storm-size threshold and dominant above
it. Hopp and McDonnell 2009 ranked the controls as storm size, slope
angle, soil depth, and bedrock permeability.

Field-based dominant-runoff-process mapping (Scherrer and Naef 2003;
Schmocker-Fackel et al. 2007) classifies sites into Hortonian overland
flow, saturation overland flow, subsurface stormflow, and deep
percolation. That is the closest thing to an empirical "which direction
dominates here" map.

The high-level synthesis arguing lateral flow matters for land-surface
models is Fan et al. 2019, Water Resources Research, "Hillslope
hydrology in global change research and Earth system modeling."
Supporting large-scale evidence: Fan et al. 2013 (Science, global
water table depth) showing where lateral groundwater convergence sets
surface wetness; Maxwell and Condon 2016 (Science) showing lateral
groundwater flow changes the transpiration fraction of ET across the
continental United States; Krakauer et al. 2014 showing lateral
groundwater exchange between grid cells matters at kilometer-scale
resolution and in low-relief humid terrain.

### Reading for the cascade question

At daily steps on sub-kilometer cells, the empirical picture says
lateral flow is thresholded and subsurface-dominated. That favors the
Dupuit-style subsurface solver of section 2 over a surface router as
the first lateral process to get right, and it argues that any lateral
scheme must carry a wetness threshold, which the fixed-fraction cascade
does not.

## 5. SUMMA's lateral machinery in detail, vs the four PRMS cascade fluxes

Read from the SUMMA source on GitHub master, 2026-09-10:
[run_oneGRU.f90](https://github.com/CH-Earth/summa/blob/master/build/source/engine/run_oneGRU.f90),
[groundwatr.f90](https://github.com/CH-Earth/summa/blob/master/build/source/engine/groundwatr.f90),
[soilLiqFlux.f90](https://github.com/CH-Earth/summa/blob/master/build/source/engine/soilLiqFlux.f90),
[computFlux.f90](https://github.com/CH-Earth/summa/blob/master/build/source/engine/computFlux.f90),
[bigAquifer.f90](https://github.com/CH-Earth/summa/blob/master/build/source/engine/bigAquifer.f90),
[qTimeDelay.f90](https://github.com/CH-Earth/summa/blob/master/build/source/engine/qTimeDelay.f90),
[mDecisions.f90](https://github.com/CH-Earth/summa/blob/master/build/source/engine/mDecisions.f90),
[var_lookup.f90](https://github.com/CH-Earth/summa/blob/master/build/source/dshare/var_lookup.f90),
and the [local attributes doc](https://summa.readthedocs.io/en/latest/input_output/SUMMA_input/).

### The only HRU-to-HRU flux

`run_oneGRU.f90`, after each HRU is solved:

```fortran
! if lateral flows are active, add inflow to the downslope HRU
if(kHRU > 0)then
 fluxHRU%hru(kHRU)%var(iLookFLUX%mLayerColumnInflow)%dat(:) = &
   fluxHRU%hru(kHRU)%var(iLookFLUX%mLayerColumnInflow)%dat(:) + &
   fluxHRU%hru(iHRU)%var(iLookFLUX%mLayerColumnOutflow)%dat(:)
else
 bvarData%var(iLookBVAR%basin__ColumnOutflow)%dat(1) = ... + sum(mLayerColumnOutflow)
end if
bvarData%var(iLookBVAR%basin__SurfaceRunoff)%dat(1) = ... + scalarSurfaceRunoff*fracHRU
bvarData%var(iLookBVAR%basin__SoilDrainage)%dat(1)  = ... + scalarSoilDrainage*fracHRU
```

`kHRU` is found by matching the attribute `downHRUindex` ("Downslope
HRU must be within the same GRU. If the value is 0, then there is no
exchange"). One receiver, no area fraction, per soil layer (m3/s),
same GRU only. `scalarSurfaceRunoff`, `scalarSoilDrainage`, and
aquifer baseflow go straight to basin totals and never touch a
neighbor. The loop runs in file order with no topological sort, so the
attributes file must list upslope HRUs first or the inflow is
misapplied.

### How that flux is computed

`groundwatr.f90`, `computBaseflow`, decision `groundwatr =
qbaseTopmodel`. Saturated thickness is stacked from the bottom of the
profile:

```
drainableWater  = mLayerDepth(iLayer)*max(0, mLayerVolFracLiq(iLayer) - fieldCapacity)/activePorosity
zActive(iLayer) = zActive(iLayer+1) + drainableWater
tran0           = kAnisotropic*surfaceHydCond*soilDepth/zScale_TOPMODEL
trTotal(iLayer) = tran0*(zActive(iLayer)/soilDepth)**zScale_TOPMODEL
trSoil(iLayer)  = trTotal(iLayer) - trTotal(iLayer+1)
mLayerColumnOutflow(iLayer) = trSoil(iLayer)*tan_slope*contourLength        ! m3 s-1
mLayerBaseflow(:) = (mLayerColumnOutflow(:) - mLayerColumnInflow(:))/HRUarea ! m s-1, net
scalarExfiltration = logF*(totalColumnInflow - totalColumnOutflow)           ! when storage nearly full
mLayerBaseflow(1)  = mLayerBaseflow(1) + scalarExfiltration
```

The lateral term is a Dupuit-style flux: transmissivity of the
saturated part (power-law conductivity decaying with depth, exponent
`zScale_TOPMODEL`) times surface slope times hillslope width.
`tan_slope` and `contourLength` are the two geometric attributes.
Exfiltration is the return flow when upslope inflow exceeds what the
column can pass on. `mDecisions.f90` forces `bcLowrSoiH = zeroFlux`,
`hc_profile = pow_prof`, and `infRateMax = topmodel_GA` with this
option. `computFlux.f90` calls `groundwatr` inside the residual
evaluation, so the flux and its derivative (`dBaseflow_dWat`) sit
inside the column's Newton iteration, but the upslope inflow is a
fixed source for that solve.

### Surface runoff generation

`soilLiqFlux.f90`, `surfaceFlx`:

```fortran
fracCap         = rootZoneLiq/(maxFracCap*availCapacity)
fInfRaw         = 1._rkind - exp(-qSurfScale*(1._rkind - fracCap))
scalarInfilArea = min(0.5_rkind*(fInfRaw + sqrt(fInfRaw**2_i4b + scaleFactor)), 1._rkind)
scalarSaturatedArea       = 1._rkind - scalarInfilArea
scalarSurfaceInfiltration = scalarInfilArea*(1._rkind - scalarFrozenArea)*min(scalarRainPlusMelt, xMaxInfilRate)
scalarSurfaceRunoff       = scalarRainPlusMelt - scalarSurfaceInfiltration
```

Saturation excess is a variable contributing area driven by root-zone
wetness and one shape parameter `qSurfScale`. No topographic index, no
upslope area. Infiltration excess is the Green-Ampt-ish `xMaxInfilRate`
from a wetting-front depth.

### GRU scale

`bigBucket` (`bigAquifer.f90`) is a per-column nonlinear reservoir,
baseflow = `aquiferBaseflowRate*(storage/aquiferScaleFactor)**
aquiferBaseflowExp`; the `singleBasin` option errors out ("not
transferred from old code base yet"). `basin__TotalRunoff` = surface +
column outflow/area + (aquifer baseflow or soil drainage), then
`qOverland` convolves it with a precomputed gamma histogram
`fracFuture` (`routingGammaShape`, `routingGammaScale`; the histogram
builder is not in qTimeDelay.f90 on master and was not located).
Stream routing is mizuRoute.

### Against the four PRMS cascade fluxes

| PRMS flux | PRMS cascade | SUMMA counterpart | crosses a unit boundary in SUMMA? |
|---|---|---|---|
| Hortonian | `upslope_hortonian`: surface water at receiver, may re-infiltrate | `scalarSurfaceRunoff` (infiltration-excess part) | no, straight to basin |
| Dunnian | `upslope_dunnianflow` into receiver capillary store | `scalarSaturatedArea` runoff + `scalarExfiltration` | no, but exfiltration is the response to upslope inflow |
| interflow | `upslope_interflow` into receiver capillary store, re-exits via `slowcoef`/`fastcoef` | `mLayerColumnOutflow` per layer, into receiver's same layers | yes, the one lateral flux |
| groundwater | `gw_upslope` (not ported in pywatershed) | `bigBucket` per column | no (`singleBasin` unimplemented) |

Contrasts that matter for design:

- Opposite emphasis. PRMS cascades surface water explicitly and buries
  subsurface transfer in re-storage. SUMMA cascades only
  saturated-subsurface water, with a physical conductance, and lets
  surface runoff leave at once. Section 4's empirical picture (lateral
  flow is subsurface, thresholded) sides with SUMMA.
- Same zero travel time. Both chains deliver within the step; neither
  holds water in transit.
- Receiver treatment. PRMS adds a lumped depth to `soil_moist`; SUMMA
  adds a layer-resolved source inside an implicit Richards solve with a
  Jacobian entry. Fill-and-spill emerges in SUMMA from `fieldCapacity`
  in `drainableWater` and the exfiltration switch; in PRMS it is
  whatever `soil_moist_max` and the coefficients produce.
- Topology. PRMS: many receivers with fractions, direct-to-segment
  links, a topological sort. SUMMA: one receiver, no fractions,
  GRU-bounded, no sort, stream delivery only via the basin total plus a
  gamma delay.
- Cost. SUMMA's lateral term is cheap because it is per-column with a
  fixed upslope source; PRMS's is a serial route-order loop.

### Can a TOPMODEL saturated-area parameterization replace cascades?

Yes for PRMS-only on hillslope-scale HRUs, replacing the Dunnian and
interflow cascade roles. No for gridded PRMS. No for GSFLOW.

What TOPMODEL proper does (Beven and Kirkby 1979; not what SUMMA does):
local deficit `S_i = S̄ + m*(λ - ln(a/tanβ)_i)`, saturated where
`S_i <= 0`, baseflow `Q_b = Q0*exp(-S̄/m)`. The upslope area `a` in
the index is the cascade graph integrated to a static number per
point. The scheme replaces dynamic unit-to-unit transfer with an
equilibrium assumption: uniform recharge, water table parallel to the
surface, exponential transmissivity. At daily steps for subsurface
flow that is no worse than PRMS's zero travel time; it is the same
assumption stated honestly.

Per flux:

- Dunnian: replaceable. PRMS already has a variable contributing area,
  `srunoff_smidx` (`ca_fraction = smidx_coef*10**(smidx_exp*smidx)`,
  capped at `carea_max`), moisture-driven with no topography. Swapping
  in a TI-distribution saturated fraction is the SIMTOP change Noah-MP
  made (Niu et al. 2005, `fsat = fsat_max*exp(-0.5*f*zwt)`).
- interflow: replaceable at HRU scale by the exponential baseflow
  `Q0*exp(-S̄/m)` in place of the gravity-reservoir `slowcoef_lin/sq`.
  That is what SUMMA's `qbaseTopmodel` is, per column.
- Hortonian: not replaceable, but daily-step Hortonian and its run-on
  re-infiltration are already the artifact flagged in section 2.
- groundwater cascade: never ported; nothing to replace.

Conditions and failures:

- The HRU must span a full TI distribution (ridge to stream). On a
  grid, each cell holds one index value and one deficit, cells do not
  exchange water, so upslope recharge never reaches the valley cell.
  Making it work means re-lumping cells into subcatchment "GRUs",
  which discards the gridded purpose. Dynamic TOPMODEL (Beven and
  Freer 2001) fixes this by routing between TI classes, which is a
  cascade again. For grids the honest alternative remains the Dupuit
  solver of section 2.
- The index is resolution-dependent (Zhang and Montgomery 1994; Wolock
  and Price 1994), and the method fails in flat, deep-water-table, or
  arid terrain (Beven 1997 critique). A static `ln(a/tanβ)` pattern
  cannot move the saturated zone between storms the way Panola's
  fill-and-spill does; the deficit threshold captures only part of
  that.
- GSFLOW: MODFLOW already resolves the lateral saturated flow and the
  water table, and soilzone receives groundwater discharge from it. A
  TOPMODEL term would double-count the subsurface lateral path.
  GSFLOW's cascades carry only surface and interflow to downslope HRUs
  and SFR reaches; the answer there is no.

Reading for the design: a process-variant pair, roughly
`PRMSRunoffTopmodel` (smidx replaced by a TI-based saturated fraction)
and `PRMSSoilzoneTopmodel` (gravity-reservoir interflow replaced by
exponential baseflow), for HRU domains delineated at hillslope scale.
Not the cascade replacement for gridded domains.

### The SUMMA-MF6 question (colleague exchange, 2026-09-10)

In a SUMMA-MF6 coupling MODFLOW takes over the saturated lateral
path. (First draft said "exactly the one lateral path SUMMA has";
section 7 corrects that: the soil-zone lateral path, PRMS's interflow
cascade, is covered only if SUMMA's `downHRUindex` is used.) What
neither covers is surface run-on (runoff from one cell re-infiltrating
on the next). Cascade-less at grid scale means every
cell's runoff reaches its reach the same day; at a daily step
overland transport across a hillslope takes far less than a day, so
only the re-infiltration is lost. Gridded surface-water models exist
(WRF-Hydro, ParFlow-CLM, MIKE SHE, HydroGeoSphere) but use explicit
sub-daily overland routers, not cascades; cascades are PRMS's
daily-step stand-in for such a router.

## 6. The channel-initiation threshold sets hillslope length, and that is what the instant-delivery schemes assume about

A channel-initiation area `A_c` sets drainage density `D`, and mean
hillslope length is `L ≈ 1/(2D)`. Every "instant delivery" scheme
(cascades, SUMMA's downslope pass, TOPMODEL's equilibrium water table)
is really the claim that water crosses, or the water table relaxes
over, `L` within one step. Scale check at a daily step:

| flux | velocity scale | `L` crossed in a day | equilibration time over `L` |
|---|---|---|---|
| overland (Hortonian, Dunnian) | 0.01 to 0.1 m/s | 1 to 10 km | any plausible `A_c` is fine |
| saturated subsurface (interflow, shallow gw) | Dupuit diffusion, `τ ≈ L² S_y/(K h)` | none; a relaxation, not a transit | `L`=100 m, `K`=10 m/d, `h`=2 m, `S_y`=0.1: ~50 d. `L`=20 m: ~2 d |

So the surface half of a cascade and the surface half of SUMMA are
insensitive to `A_c` at daily steps. The subsurface half is sensitive
as `L²`: the "hillslope water table is in equilibrium with today's
storage" assumption behind both re-storage cascades and TOPMODEL is
only defensible when the network is dense enough that hillslopes are
tens of meters, or `K` is macropore-high. That is the condition under
which TOPMODEL is reported to work (humid, steep, shallow soils), read
the other way round.

### Where `A_c` shows up in each model

- PRMS/NHM without cascades. `A_c` defines the segments, and the HRUs
  are the segment catchments, so `L` is the HRU scale. All delay is in
  `gwflow_coef`, `slowcoef_*`, `fastcoef_*`. Drainage density is not a
  parameter; it is baked into what those coefficients mean. Change
  `A_c` and the calibrated coefficients belong to a different model.
- Gridded PRMS with cascades. `A_c` sets chain length `N = L/dx`.
  Transit is zero regardless, but the subsurface delay is `N`
  re-storages, so it scales with cell count, not with `L` physically.
  Halve `dx`, double the delay for the same hillslope. Denser streams
  shorten chains and hide the artifact; in the limit `A_c = dx²` every
  cell touches a reach and cascades reduce to no-cascade. This is the
  model where the threshold matters most, and for the worst reason.
- SUMMA `qbaseTopmodel`. `L` is explicit: `contourLength` is "width of
  a hillslope parallel to a stream", and outflow per unit area is
  `T tanβ contourLength/HRUarea = T tanβ/L`. Drainage density is a
  physical input to the flux. Everything downstream of the column is
  folded into `routingGammaScale` and the mizuRoute network, whose own
  threshold defines the GRUs, so it is NHM-like there.
- TOPMODEL proper. The index `ln(a/tanβ)` is computed with flow paths
  truncated at channel cells, so the distribution and hence the
  saturated fraction depend on `A_c` directly (Quinn, Beven and Lamb
  1995, Hydrological Processes, "The ln(a/tanβ) index: how to
  calculate it and how to use it within the TOPMODEL framework"). The
  `L²` relaxation argument above is the physical content of its
  steady-state assumption.
- GSFLOW / MF6. MODFLOW's lateral flow has no equilibrium assumption,
  so `A_c` stops mattering for groundwater. It still decides which
  physics the same surface water gets: a reach means SFR streambed
  exchange; a downslope cell means cascade then UZF infiltration.
- Dupuit or diffusive-wave router (section 2). No equilibrium
  assumption. `A_c` only places the boundary condition. The one
  approach where the threshold is just a geometric fact.

### A wrinkle worth naming

Zeroth-order basins are the hollows where saturation overland flow and
subsurface stormflow are generated, and the channel network expands
into them in wet seasons. Choosing a low `A_c` turns that dynamic
source area into a fixed channel: the saturated-area mechanism gets
replaced by routing. A dense network can therefore substitute for both
cascades and TOPMODEL, at the cost of freezing the network extent.

## 7. What GSFLOW actually stacks, and what a SUMMA-MF6 coupling would lose

Read from the GSFLOW source
([rniswon/gsflow_v2](https://github.com/rniswon/gsflow_v2):
`GSFLOW/src/gsflow/gsflow_prms.f90`, `gsflow_prms2mf.f90`,
`gsflow_mf2prms.f90`, `GSFLOW/src/prms/soilzone.f90`,
`GSFLOW/src/modflow/gwf2uzf1_NWT.f`) and its Sagehen example
(`GSFLOW/data/sagehen/`), 2026-09-10. USGS pubs blocked fetches, so the
manual (Markstrom et al. 2008, TM 6-D1) is not quoted here.

### The stack, top down

All of PRMS soilzone still runs in GSFLOW mode: capillary reservoir
(root zone, ET), gravity reservoir, preferential-flow reservoir, and
the `slowcoef_*`/`fastcoef_*` interflow and Dunnian logic. What
changes is who the gravity reservoir drains to and who feeds it back:

```fortran
! gsflow_prms2mf.f90: gravity drainage -> UZF infiltration, per HRU x cell intersection
Cell_drain_rate(icell) = Cell_drain_rate(icell) + Sm2gw_grav(j)*Gvr2cell_conv(j)
! gsflow_mf2prms.f90: UZF/MODFLOW seepage at land surface -> back into the gravity reservoirs
Gw2sm_grav(i) = SEEPOUT(Gwc_col(Gvr_cell_id(i)), Gwc_row(Gvr_cell_id(i)))*Mfq2inch_conv(i)
! soilzone.f90: cascade to a stream segment
Strm_seg_in(j) = Strm_seg_in(j) + DBLE((Slowflow+Preflow+Dunnian)*Cascade_area(k,Ihru))*Cfs_conv
! gsflow_prms.f90
IF ( GSFLOW_flag==ACTIVE .AND. Call_cascade==OFF ) THEN
  PRINT *, 'ERROR, GSFLOW requires that PRMS cascade routing is active'
```

The gravity reservoir is split into one reservoir per HRU-cell
intersection (`nhrucell`). Each drains vertically into its cell's UZF
column, capped by UZF's vertical `VKS`; what UZF will not take stays in
the gravity reservoir and leaves sideways as PRMS interflow. `gwflow`
is replaced by MODFLOW, `strmflow` by SFR, which takes `Strm_seg_in` as
reach inflow. Groundwater that reaches land surface comes back into the
soil zone, overfills it, and leaves as Dunnian runoff through the
cascades. All three lateral cascade fluxes remain PRMS's job in GSFLOW.

### UZF is not 2D

The UZF1 header states it: a one-dimensional vertical kinematic wave
per cell, "lateral unsaturated flow is neglected". It is the vadose
zone below the root zone down to the water table, with no lateral
term. Lateral subsurface flow in GSFLOW is therefore exactly two
things: PRMS interflow cascades in the soil zone, and MODFLOW saturated
flow. Nothing in between.

### Why the vertical stack looks doubled

PRMS-only sends `soil_to_gw` and `ssr_to_gw` straight to a linear
groundwater reservoir; there is no vadose zone. GSFLOW keeps PRMS's
thin, ET-active, calibrated soil zone and inserts UZF beneath it for
the travel time through a thick vadose zone (Sagehen's ridges), then a
real aquifer. The overlap is only that both the gravity reservoir and
UZF hold water in vertical transit, and the `VKS` cap is the seam that
decides who holds it. The lateral physics is not doubled: the soil
zone is lateral-by-cascade, UZF is vertical only, the aquifer is
lateral-by-Darcy.

### Sagehen numbers

| item | value |
|---|---|
| HRUs / cascade links | 128 / 317 (266 to an HRU, 51 to a segment via `hru_strmseg_down_id`) |
| MODFLOW grid | 2 layers, 73 x 81 = 5913 cells (`ngwcell`; the whole grid, not the active count), 90 m cells; basin about 27 km2 so roughly 3300 active |
| scale | HRU mean about 0.21 km2, about 26 cells, about 5 cells across |
| HRU-cell gravity reservoirs | 4691 |
| SFR | 15 segments, 201 reaches |
| UZF `IRUNFLG` | 0 (UZF does no stream routing; PRMS cascades do it) |

(`cascade_flg = 0` in `prms.params` is the many-to-one/one-to-one
parameter, not the control-file `cascade_flag`.)

The MODFLOW 6 example
([ex-gwf-sagehen](https://modflow6-examples.readthedocs.io/en/latest/_examples/ex-gwf-sagehen.html))
removes PRMS entirely: infiltration is a daily time series times an
altitude factor, ET is an extinction depth, runoff is UZF rejected
infiltration plus DRN groundwater discharge, and MVR carries both to
the nearest downgradient reach. No soil zone, no interflow, no
cascades.

### Consequence for a SUMMA-MF6 coupling

Section 5's line "MODFLOW takes over exactly the lateral path" was
overstated. MF6 takes the saturated path. The soil-zone lateral path
(PRMS interflow cascades) is covered by nobody unless SUMMA's
`downHRUindex` is used, and that is precisely the flux it passes
(section 5). So the ask is not "cascades" but "use `downHRUindex`
within GRUs", existing SUMMA code. Surface run-on (runoff from one
unit re-infiltrating on the next) is what PRMS cascades provide, via
`compute_infil` + `perv_comp` on `Upslope_hortonian` at every receiving
HRU, and what neither SUMMA nor MF6 has; at daily steps it is the
cheaper loss (section 6).
The palatability problem with the cascade-less MF6 Sagehen example is
the missing interflow, not the missing cascades.

## 8. GSFLOW time stepping: PRMS inside MODFLOW's solver iteration

Read from the local GSFLOW 2.4.0 source (`~/usgs/gsflow/gsflow_v2.4.0`,
the fork pywatershed builds its GSFLOW binaries from), 2026-09-10;
cross-checked against rniswon/gsflow_v2 on GitHub, which differs only
in the order of calls inside the iteration loop (the local copy is
taken as authoritative below). Line numbers are 2.4.0's.

### Clock

One MODFLOW time step is one PRMS day, enforced, and `MFNWT_RUN` is
called once per day (`src/gsflow/gsflow_modflow.f`):

```fortran
1526      IF ( ABS(Timestep_seconds-DELT*Mft_to_sec)>NEARZERO ) THEN     ! MFNWT_RDSTRESS
1576 9003 FORMAT (' Time steps must be equal: PRMS dtsec = ', F0.4, ...
          IF(AFR) KSTP = KSTP + 1                                        ! MFNWT_RUN
          KKSTP = KSTP
```

Stress-period boundaries become Julian days from `modflow_time_zero`
plus `PERLEN` (`SET_STRESS_DATES`); stress periods before the PRMS
start date are stepped through without solving (`Modflow_skip_time`);
only stress period 1 may be steady state ("only first time step can
be SS"), and it runs through the same `MFNWT_RUN` with
`Steady_state = 1`. Sagehen: SP1 steady, SP2 transient with 5844
one-day steps.

### Per day, outside the iteration (`src/gsflow/gsflow_prms.f90`)

`prms_time` (`src/prms/sm_prms_time.f90` 43-70) advances the day, sets
`timestep_start_flag = ACTIVE`, reads the data line, and takes the
day-start snapshot every iterating module will rewind to:

```fortran
Timestep = Timestep + 1
timestep_start_flag = ACTIVE
...
It0_soil_moist = Soil_moist
It0_soil_rechr = Soil_rechr
It0_ssres_stor = Ssres_stor
It0_slow_stor = Slow_stor
It0_pref_flow_stor = Pref_flow_stor
It0_pkwater_equiv = Pkwater_equiv
IF ( GSFLOW_flag==ACTIVE ) It0_gravity_stor_res = Gravity_stor_res
It0_hru_impervstor = Hru_impervstor
It0_hru_intcpstor = Hru_intcpstor
IF ( PRMS_land_iteration_flag==CANOPY ) THEN
  It0_intcp_transp_on = Intcp_transp_on
  It0_intcp_stor = Intcp_stor
ENDIF
It0_imperv_stor = Imperv_stor
```

Then climate, potet, transp run once. Which land-surface modules run
once here versus inside the loop is set by the control parameter
`PRMS_land_iteration_flag` (gsflow_prms.f90 457-461, 1033):

```
! PRMS_land_iteration_flag: 0 = soilzone only; 1 = srunoff and soilzone;
!                           2 = intcp, snowcomp, srunoff, and soilzone in iteration loop
! soilzone is always in MODFLOW iteration loop
```

Default 0: canopy, snow, and srunoff run once per day before
`MFNWT_RUN`; only soilzone iterates.

### Inside the day: the loop (`gsflow_modflow.f`)

```
628    Szcheck = OFF
629    IF ( gsflag==ACTIVE ) Szcheck = ACTIVE
631    KITER = 0
641    DO WHILE (ITREAL2.LT.MXITER)
642      KITER = KITER + 1
652      CALL GWF2BAS7FM(IGRID)                      ! flow-package formulation
694      IF ( (Szcheck==ACTIVE .AND. Model==GSFLOW) ... ) THEN
697        IF ( PRMS_land_iteration_flag==CANOPY ) intcp, snowcomp, glacr
706        IF ( PRMS_land_iteration_flag>0 ) retval = srunoff()
710        retval = soilzone()
714        retval = gsflow_prms2mf()                 ! Sm2gw_grav -> FINF (binned); Strm_seg_in -> STRM(12,k)
715        Sziters = Sziters + 1
717        IF ( KKITER==Mxsziter ) Szcheck = OFF     ! stop calling PRMS in iteration loop
718      ELSEIF ( iss==0 ) THEN
719        IF ( KKITER==Mxsziter+1 ) Stopcount = Stopcount + 1
723      CALL GWF2UZF1FM(...)                        ! uses this iteration's FINF
731      CALL GWF2SFR7FM(...)                        ! uses this iteration's reach inflow
813      CALL GWF2NWT1FM(KKITER,ICNVG,...)           ! solve heads
849      IF (ICNVG.EQ.1) GOTO 33                     ! converged: exit, no hand-back
850      IF ( Szcheck==ACTIVE .AND. GSFLOW_flag==1 ) retval = gsflow_mf2prms()   ! SEEPOUT -> Gw2sm_grav
       END DO
```

Iteration k: rewind the land surface to the day-start snapshot, run it
with today's forcing and the seepage MODFLOW produced at solve k-1
(`Gw2sm_grav`, zero on k=1), hand `FINF` and reach inflow to UZF and
SFR, solve heads. If not converged, map the new `SEEPOUT` back and go
again. A lagged Picard scheme. On convergence the day ends with
soil-zone states computed from seepage k-1 and heads from solve k.

### Convergence

MODFLOW's `ICNVG` (NWT head/flux closure) is the only test. There is
no soil-zone tolerance: `grep -rni szconverge src` finds nothing in
2.4.0 (the 2008 manual's `szconverge` is gone). `mxsziter` caps how
many iterations call PRMS; after it PRMS is frozen and MODFLOW keeps
iterating on stale fluxes. Its default (gsflow_modflow.f 405-410):

```fortran
! maximum MF iterations, which is a good practice using NWT and cells=nhru
IF ( Mxsziter<1 ) Mxsziter = MXITER
```

`Stopcount` counts days that hit the cap; the run summary prints
MODFLOW iterations per day and PRMS calls per day.

### The rewind, per module

```fortran
! src/prms/soilzone.f90
803 ! It0 variables used with MODFLOW integration to save iteration states.
805   IF ( timestep_start_flag == ACTIVE ) THEN
809     Gw2sm_grav = 0.0 ! dimension nhrucell
        IF ( PRMS_land_iteration_flag==OFF ) THEN
          ! computed in srunoff
          It0_sroff = Sroff
          It0_hru_sroffp = Hru_sroffp
          It0_hortonian_flow = Hortonian_flow
815       It0_strm_seg_in = Strm_seg_in
        ENDIF
818     timestep_start_flag = OFF
      ELSE
820     Soil_moist = It0_soil_moist
        Soil_rechr = It0_soil_rechr
        Ssres_stor = It0_ssres_stor
        Slow_stor = It0_slow_stor
        IF ( Pref_flag==ACTIVE ) Pref_flow_stor = It0_pref_flow_stor
826     Gravity_stor_res = It0_gravity_stor_res
! src/prms/srunoff.f90
659   IF ( Kkiter>1 ) THEN
        IF ( PRMS_land_iteration_flag>0 ) THEN
          Imperv_stor = It0_imperv_stor
          Soil_moist = It0_soil_moist
          Soil_rechr = It0_soil_rechr
! src/prms/intcp.f90
291     IF ( Kkiter>1 ) THEN
          Intcp_stor = It0_intcp_stor
          Hru_intcpstor = It0_hru_intcpstor
```

`Strm_seg_in` is rewound to srunoff's Hortonian value each iteration
and soilzone re-adds interflow and Dunnian on top. The contributing-
area functions read the antecedent copies (`srunoff.f90` 1259, 1267:
`smidx = It0_soil_moist(Ihru) + 0.5*Ptc`), so runoff generation does
not drift with the iteration count. Cascades are re-executed inside
srunoff and soilzone on every iteration, part of why GSFLOW days are
expensive.

What is iterated: heads, UZF, SFR, gravity-reservoir drainage,
soil-zone storages, Dunnian and interflow to reaches. Canopy and snow
only with flag 2. Climate, potential ET, transpiration: never.

### Two lines in the snowcomp save block look like bugs (flag 2 only)

`src/prms/snowcomp.f90`, iteration-1 save branch under
`PRMS_land_iteration_flag==CANOPY`:

```fortran
974          It0_albedo = It0_albedo
986          It0_iso = Iso
988          It0_iso = Lso          ! overwrites the Iso copy; It0_lso is never assigned
```

Under flag 2 the rewind restores `Albedo` from a never-updated copy
and `Iso` from `Lso`. Flag 0, the default, is unaffected. Worth
reporting upstream if pywatershed ever runs GSFLOW with flag 2.

### Run-on, for the record

PRMS cascades DO provide surface run-on: cascaded Hortonian water is
added to `Infil` at the receiving HRU and re-partitioned by the
contributing-area function (`srunoff.f90` 1057-1064, `perv_comp`
1242-1283); only `ca_fraction` of it continues as runoff. The direct
route exists only for links that point at a segment
(`hru_strmseg_down_id`; 51 of 317 at Sagehen). SUMMA's
`scalarSurfaceRunoff` and MF6's UZF have no such stop.
