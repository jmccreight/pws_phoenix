---
inclusion: manual
---

# Note: Is the DAG solve in Muskingum-style stream routing necessary?

Status: analysis / literature scan. No code written, no experiments run.
Originating conversation: asked while the `pywatershed` project was open, but the
question is general and the intended home is a different project.
Date recorded: see file mtime.

## The question as posed

Muskingum-like channel routing (e.g. `PRMSChannel` in pywatershed) solves on a
directed acyclic graph with a topological ordering at each timestep. What if
instead you took smaller timesteps and set each element's upstream inflow to the
sum of upstream outflows from the *prior* substep? Then there is no ordering
constraint and the solve is embarrassingly parallel. Some parameters of the
original method would presumably need to change. Is the DAG solve necessary, a
historical relic, or still genuinely relevant?

## Bottom line

The DAG ordering is not a relic, but it is not fundamental either. It is the
exact, cheap way to solve an implicit-in-space coupling: under
upstream-to-downstream ordering the per-substep network equations form a lower
triangular linear system, and the topological sweep *is* forward substitution.
O(N), direct, no iteration, no tolerance.

The lagged (explicit) alternative is stable and mass-conserving. It fails only
in phase and amplitude, in a characterizable way. The real obstacle is that the
Δt needed to control the lag error is coupled to the Δt the Muskingum
coefficients permit, and those two constraints pull in opposite directions.

This has been explored in the literature, including almost exactly the proposed
formulation. See references below.

## What creates the dependency (two distinct things)

Using pywatershed as the concrete example — separate these, because they have
different fixes:

1. **The `c0` term.** The update is
   `outflow_ts[j] = inflow_ts[j]*c0 + inflow_ts_prev[j]*c1 + outflow_ts[j]*c2`,
   where `inflow_ts[j]` is the *current* substep inflow. This is the implicit
   coupling proper.
2. **The inflow accumulation.** `seg_upstream_inflow[to_seg] += outflow_ts[jseg]`
   executes inside the same substep iteration. So even where `c0` has been
   clamped to zero, building `inflow_ts[j]` still requires the current substep's
   upstream outflow. The ordering is needed for both.

Point 2 is easy to miss and matters: clamping `c0` does not by itself free you
from the ordering.

## Why lagging is safe in some respects and not others

Replacing the current-substep inflow with the previous one introduces error
`c0*(I_t - I_{t-1}) ~= c0 * Δt * dI/dt`. First order in Δt, where the box
scheme was second order.

- **Phase / lag error.** Each hop adds roughly Δt of spurious delay. Along a
  path of topological depth D the accumulated lag is ~ D*Δt. Per reach the
  relative travel time error is Δt/K, which sounds mild, but at the outlet what
  matters is D*Δt against the hydrograph time of rise. Daily domain, D in the
  hundreds, Δt = 1 h gives days of spurious delay. Δt of minutes begins to be
  defensible.
- **Amplitude error.** Changing Δt changes the scheme's numerical diffusion, so
  peak attenuation moves even where timing is acceptable.
- **Stability is NOT the problem.** Each reach is a stable linear filter
  (|c2| < 1) and pure delays do not destabilize a cascade of stable filters.
  It will not blow up.
- **Mass is NOT the problem.** Volume is conserved either way. In pywatershed
  `seg_stor_change = (seg_inflow - seg_outflow) * dt` closes by construction.
  The error is purely in *when* and *how peaked*, never *how much*.

That single, isolated failure mode is what makes this a clean experiment.

## The parameter coupling — the crux

The Muskingum coefficient positivity window is

```
2*K*x  <=  Δt  <=  2*K*(1 - x)
```

with the standard coefficients (Δt written for the substep, `ts` in the code):

```
d  = K - K*x + 0.5*Δt
c0 = (-K*x + 0.5*Δt) / d
c1 = ( K*x + 0.5*Δt) / d
c2 = ( K - K*x - 0.5*Δt) / d
```

So `c0 < 0  <=>  Δt < 2*K*x` (labelled "Long travel time" in the code) and
`c2 < 0  <=>  Δt > 2*K*(1-x)` ("Short travel time").

Note the **lower** bound on Δt. Shrinking Δt with K and x fixed drives `c0`
negative. The per-segment `tsi`/`ts` lookup table (Δt rounded down to an even
divisor of 24 h, keyed on K) is in effect a device for landing Δt inside that
window for each segment. It is not arbitrary.

Consequences:

- "Just use a small uniform Δt" pushes every segment below the lower bound. The
  only way back inside the window is x -> 0, which is the pure linear reservoir
  limit: maximum attenuation, no wedge storage, none of the translation
  behaviour x existed to provide. (Equivalently a Kalinin-Milyukov / Nash
  cascade of linear reservoirs.)
- Given how `ts` relates to K in that table (roughly K/2 to K/1.5), `c0 < 0`
  bites around x >~ 0.25. Typical PRMS `x_coef` of 0.2 stays positive; 0.5 gets
  clamped. **Domain-specific — must be checked, not assumed.**
- Where `c0` has already been clamped to 0, the routing update is already
  explicit; only the accumulation (point 2 above) still needs the ordering.
- If you move to Muskingum-Cunge, where K and x derive from Δx and hydraulics
  rather than calibration, the operative constraint becomes the Courant number
  C = c*Δt/Δx. Accuracy is best near C ~ 1. Dropping Δt with reach length fixed
  drives C well below 1. Keeping C ~ 1 requires splitting reaches, which
  *increases* depth D, which multiplies the accumulated lag error you were
  trying to shrink. **This circularity is the core tension and the reason the
  problem does not reduce to "pick Δt small enough".**

## Do this before breaking the DAG

The DAG contains a great deal of unused parallelism. `nx.topological_generations`
gives level sets; within a level every element is independent and can be
vectorized with **zero** accuracy cost. Dendritic networks are very wide near the
headwaters, and depth scales roughly with mainstem length (Hack's law puts that
near A^0.57), so concurrency on the order of N/depth is available exactly.

Any approximate scheme has to beat that baseline to be worth its error. Measure
the wavefront speedup first.

Where lagging genuinely wins is a regime wavefronts cannot reach: GPU kernels
where level-synchronous execution is awkward, or distributed runs where the
level-set barrier forces communication every substep. For those, the
halo/overlap approach (below) is the principled version — tunable error rather
than uncontrolled error.

## Framings worth keeping in view

- **It is a DAG because the model assumes no backwater.** Drop that (dynamic
  wave, reservoir influence, tidal or downstream control) and you have upstream
  *and* downstream coupling and no DAG regardless of Δt. Since small Δt is
  precisely the regime where the lumped-storage Muskingum abstraction is least
  justified, "should the physics change too?" belongs alongside the parallelism
  question.
- **Localization / truncated-series view.** Casting network Muskingum as a
  matrix system, the inverse can be expanded as a Neumann-type series whose
  terms correspond to influence propagating successively further upstream. Those
  terms decay, which is the formal statement of "distant reaches are nearly
  independent within one update." This is the theoretical basis for both halo
  methods and the David et al. quantification below, and it suggests a
  middle path: exact within a k-hop neighbourhood, lagged beyond it.
- **Keep the two knobs separate.** Substep size Δt and the coupling scheme
  (implicit / lagged / k-hop) are independent choices. Conflating them makes
  experiment results uninterpretable.

## Literature

Most directly relevant, the RAPID line of work (Cédric David and colleagues):

- *Quantification of the upstream-to-downstream influence in the Muskingum
  method and implications for speedup in parallel computations of river flow*,
  WRR 2013, doi:10.1002/wrcr.20250.
  https://escholarship.org/uc/item/6gv9t32c
  This is the question, asked and quantified: how far upstream does
  within-timestep influence actually reach, and what does that buy in parallel.
- *Enhanced fixed-size parallel speedup with the Muskingum method using a
  trans-boundary approach and a large subbasins approximation*, WRR 2015,
  doi:10.1002/2014WR016650.
  https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2014WR016650
  Builds on what they describe as the relative independence of distant reaches
  in the update step.
- *River network routing on the NHDPlus dataset* (RAPID), David et al., J.
  Hydrometeorology 2011.
  https://journals.ametsoc.org/downloadpdf/view/journals/hydr/12/5/2011jhm1345_1.pdf
  The other route entirely: cast network Muskingum as a matrix equation and hand
  it to a parallel linear solver (PETSc). Keeps implicitness exactly, gets
  parallelism from the linear algebra rather than the graph.

The proposed scheme, with a mitigation:

- *A Stream-Order Family and Order-Based Parallel River Network Routing Method*,
  Water 2024. https://www.mdpi.com/2073-4441/16/14/1965
  Order-based wavefront parallelism, then deliberately relaxes the
  upstream-downstream dependency along the longest flow paths to split the
  network into independent subnetworks, adding *halo reaches* to absorb the
  error from inexact inflows. Classic domain-decomposition overlap on a river
  network; halo width is the accuracy knob.

Numerical accuracy criteria:

- Ponce & Theurer, *Accuracy Criteria in Diffusion Routing*, ASCE 1982.
  https://ascelibrary.org/doi/10.1061/JYCEAJ.0005872
  Resolution requirements in terms of Courant and cell Reynolds numbers.
- Ponce, Muskingum-Cunge with variable parameters and the amplitude/phase
  "portraits" (after Cunge 1969). https://ponce.sdsu.edu/
  Useful for reasoning about where in (C, cell Reynolds, x) space a scheme is
  convergent.

Implementations worth reading:

- t-route (NOAA/NextGen), reach and subnetwork decomposition:
  https://github.com/awlostowski-noaa/t-route
- river-route, numba + sparse-matrix Muskingum-family routing:
  https://github.com/rileyhales/river-route
- Differentiable Muskingum-Cunge with learned Manning n and geometry
  (delta-MC, hydroDL2), relevant if parameters are to be re-learned for a new Δt
  rather than re-derived:
  https://repository.library.noaa.gov/view/noaa/63586/noaa_63586_DS1.pdf

Content from these sources was rephrased for compliance with licensing
restrictions. Verbatim reproduction was avoided.

## Proposed experiments

Ordered by information per unit effort.

1. **Diagnostics only, no new physics.** Distributions of K (`_Kcoef`),
   `_ts`/`_tsi`, and `_c0`/`_c1`/`_c2` on real domains. What fraction of
   segments have `c0` clamped to zero? Then the topological level profile: depth
   D, width per level, width distribution. This simultaneously tells you how
   much implicit coupling actually exists and how much exact parallelism is
   available. Cheap. Do this first; it conditions everything else.
2. **Lagged vs. DAG at fixed parameters.** Δt = 1 h, then 1/2, 1/4, 1/8 h.
   Metrics: peak error, timing error at outlet *and* interior gauges, volume
   closure. Falsifiable prediction: O(Δt) convergence to the DAG solution, plus
   a systematic positive lag of approximately D*Δt.
3. **Lag compensation.** Set K_eff = K - Δt (and/or adjust x) and see how much
   phase error that recovers. Distinguishes "reparameterization problem" from
   "structural problem".
4. **k-hop / halo variant.** Error as a function of halo width, following the
   Water 2024 approach. Gives the tunable-error curve.
5. **Exact wavefront baseline.** Level-synchronous vectorized solve. This is the
   speedup number every approximate scheme must beat.

Prior (stated as a prior, not a result): experiments 1 and 2 will show the lag
error is dominated by mainstem topological depth, and that the Δt required to
control it sits far enough below the coefficient positivity floor that the
scheme is no longer Muskingum in any meaningful sense — at which point solving
the diffusive wave directly is the more honest option.

## pywatershed specifics (for anyone returning to that repo)

Repo root: `/Users/jmccreight/usgs/pywatershed`

- `pywatershed/hydrology/prms_channel.py`
  - `PRMSChannel._initialize_channel_data` — builds `networkx.DiGraph`,
    `nx.topological_sort` at ~line 262, derives `_Kcoef`, the `ts`/`tsi` lookup
    table, and `_c0/_c1/_c2`. Coefficient clamps at ~lines 364-374
    ("Short travel time" / "Long travel time").
  - `PRMSChannel._muskingum_mann_numpy` — the solver. Serial loop over
    `segment_order` inside a hardcoded `for ihr in range(24)`. The
    cross-segment write is `seg_upstream_inflow[to_seg] += outflow_ts[jseg]` at
    ~line 617.
  - `calc_method` is `"numba"` by default, jitted with `parallel=False`.
- `pywatershed/hydrology/prms_channel_flow_graph.py` — **the good place to
  experiment.** `PRMSChannelFlowNode` already exposes an `outflow_substep`
  property, and the sweep contract is
  `calculate_subtimestep(ihr, inflow_upstream, inflow_lateral)`. A lagged
  variant is therefore a change in the *FlowGraph driver* (feed the previous
  substep's `outflow_substep`), not a change to node physics. That gives both
  schemes side by side on identical parameters. Duplicate coefficient
  derivation lives in `PRMSChannelFlowNodeMaker._init_data` (clamps ~lines
  317-326) — note it lacks the `_adjust_parameters` guard that
  `prms_channel.py` has.
- Also present: `pass_through_flow_node.py`, `obsin_flow_node.py`,
  `source_sink_flow_node.py`, `starfit_source_sink_flow_node.py` — so the
  FlowGraph node abstraction is already exercised by several node types.
- Example notebook referenced in docstrings:
  `examples/06_flow_graph_starfit.ipynb`.

### Important: a diffusive-wave reference already exists in this repo

`pywatershed/utils/mmr_to_mf6_dfw.py` (plus `mmr_to_mf6_mmr.py`) converts
Muskingum-Mann routing setups to MODFLOW 6 CHF/DFW — channel flow with the
diffusive wave equation. Supporting material:

- `examples/07_mmr_to_mf6_chf_dfw.ipynb`
- `autotest/test_mmr_to_mf6_dfw.py`

This matters a great deal for the experiment plan. The note's own conclusion is
that pushing Δt small enough to make a lagged scheme accurate takes you outside
the regime where Muskingum is meaningful, and that solving the diffusive wave
directly is then the more honest option. **That comparison path is already
built.** So:

- There is an existing route to a higher-fidelity reference solution, rather than
  only DAG-Muskingum as the reference. Experiment 2 can be scored against DFW
  instead of merely against the DAG answer, which is a much stronger test — it
  separates "differs from the legacy scheme" from "differs from the better
  physics".
- MF6 DFW is an implicit solve with its own solver and parallel story, which is
  the natural comparison point for the "just change the physics" branch.
- Whoever picks this up should read `mmr_to_mf6_dfw.py` and the notebook before
  designing anything, and check whether the domains used in
  `test_mmr_to_mf6_dfw.py` are suitable test cases (they come pre-wired with
  both a Muskingum and a diffusive-wave configuration).

Not relevant despite the name: `examples/parallel/` is a parameter-ensemble
sweep (`run_params_000` ... `run_params_010`), not spatial parallelism.

## Verification status — read this before relying on the above

- Code claims (structure, line numbers, `c0` clamp behaviour, FlowGraph API)
  were read directly from the repo and are reliable.
- The algebra (coefficient formulas, positivity window, first-order error term)
  is derivable from the code as written and was checked by hand.
- Literature claims come from **search result snippets and abstracts only**.
  Full-text fetches returned HTTP 403 for the Wiley, AMS, MDPI and eScholarship
  items. Titles, DOIs and URLs are solid; the summaries of *what each paper
  concludes* should be confirmed against the actual papers before being cited or
  built upon. In particular the David et al. 2013 quantitative result — how many
  reaches upstream the within-timestep influence remains significant — was NOT
  retrieved and is the single most valuable missing number.
- The Hack's-law depth scaling and the N/depth concurrency estimate are
  back-of-envelope, not measured on any real network. Experiment 1 replaces them
  with facts.
- No experiment was run. Every performance and error statement above is
  prediction.
