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
be chained hillslope-style through a `downHRUindex` attribute. Surface
runoff and saturated-zone outflow from an upslope HRU become inflow to
the downslope HRU within the same step: the same instantaneous cascade
idea as PRMS, with a single receiver and no area fractions. It is
rarely used; most setups make each GRU one HRU.

At the GRU scale, lateral drainage is a parameterization, not a
transport: the TOPMODEL-style baseflow option (power-law
transmissivity, saturated fraction from a topographic index) or a
simple bucket. Runoff leaving a GRU goes to mizuRoute, which routes
only streamflow. Hillslope water never crosses a GRU boundary.

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

Caveat: the GRU-independence design and the mizuRoute split are
solid; the exact attribute name and what fluxes the within-GRU chain
passes are from memory of SUMMA v2/v3 and should be checked against
the SUMMA docs before citing.
