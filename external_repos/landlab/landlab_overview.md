# Landlab Overview: Grids, Networks, and Compatibility with pywatershed

_Based on Landlab documentation, source inspection, and the pywatershed `incarnations/xr`
prototype. Written to inform decisions about Landlab interoperability._

---

## 1. What Is Landlab?

Landlab is a Python toolkit for building 2D (and quasi-1D) earth-surface process models.
Its central design principle is that a **ModelGrid** object serves as the shared namespace
for all model state. Components read and write named fields on the grid
(`grid.at_node["topographic__elevation"]`, `grid.at_link["flow_depth"]`, etc.) and are
otherwise decoupled from each other. The grid is the message bus.

This is fundamentally different from pywatershed's approach, where state is shared via
numpy buffer references between Process objects, with no global grid object. That
difference is the crux of the compatibility question.

---

## 2. Grid Taxonomy

Landlab provides seven grid classes, all inheriting from `ModelGrid`:

| Class                 | Node arrangement        | Has cells? | Has patches? | Primary use                         |
| --------------------- | ----------------------- | ---------- | ------------ | ----------------------------------- |
| `RasterModelGrid`     | Regular rectangular     | Yes        | Yes          | DEMs, 2D diffusion, overland flow   |
| `VoronoiDelaunayGrid` | Arbitrary (x,y) points  | Yes        | Yes          | Irregular unstructured 2D domains   |
| `FramedVoronoiGrid`   | Perturbed rectangular   | Yes        | Yes          | Quasi-irregular 2D                  |
| `HexModelGrid`        | Hexagonal               | Yes        | Yes          | Isotropic 2D, cellular automata     |
| `RadialModelGrid`     | Concentric circles      | Yes        | Yes          | Radially symmetric problems         |
| `NetworkModelGrid`    | Ad libitum (x,y) points | **No**     | **No**       | 1D branching channel/reach networks |
| `IcosphereGlobalGrid` | Spherical (icosahedral) | Yes        | Yes          | Global-scale models                 |

The key topological elements are:

- **Nodes** -- scalar state (elevation, water depth, storage, etc.)
- **Links** -- directed edges between nodes; vector quantities (flux, gradient)
- **Cells** -- interior polygons (Voronoi or square); area-weighted quantities
- **Faces** -- shared edges between cells; flux boundaries
- **Patches** -- dual polygons (Delaunay triangles or squares); complement to cells

`NetworkModelGrid` is the structural outlier: it has only nodes and links, no cells or
faces. This is appropriate for a 1D river network where there is no areal extent to
discretize.

---

## 3. NetworkModelGrid: The Channel Network Grid

`NetworkModelGrid` is the grid type most relevant to pywatershed because it represents
a **sparse, branching, directed 1D network** -- the closest Landlab analog to a
reach-based or HRU-based routing network.

### Construction

```python
from landlab.grid.network import NetworkModelGrid

y_of_node = (0, 100, 200, 200, 300)
x_of_node = (0, 0, 100, -50, -100)
nodes_at_link = ((1, 0), (2, 1), (3, 1), (3, 4))

grid = NetworkModelGrid((y_of_node, x_of_node), nodes_at_link)
```

- **Nodes** are junctions or headwaters/outlets -- any topological break in the network.
- **Links** are reaches -- each link connects two nodes and can carry reach-scale fields
  (`reach_length`, `channel_width`, `flow_depth`, etc.).
- The connectivity (`nodes_at_link`) is supplied explicitly by the user, not inferred
  from spatial proximity. This means arbitrary topologies -- including bifurcations and
  confluences -- are representable.

### Fields

Fields can be attached at nodes or links:

```python
grid.at_node["topographic__elevation"] = np.array([...])
grid.at_link["reach_length"] = np.array([...])
```

There are no cells on a `NetworkModelGrid`, so `grid.at_cell` does not apply.

### Flow direction

`FlowDirectorSteepest` can be run on a `NetworkModelGrid` to determine upstream/downstream
orientation of each link based on nodal elevation. This is a prerequisite for most
routing components.

### Relationship to raster grids

Landlab provides tooling to **derive** a `NetworkModelGrid` from a `RasterModelGrid`
(via flow accumulation and channel extraction) and to import one from a shapefile or
NHDPlus HR dataset. This means a physically-based network topology can be extracted from
a DEM and then used to drive a channel-scale model.

### Relevant components that operate on NetworkModelGrid

- `NetworkSedimentTransporter` -- Lagrangian bedload sediment transport through a network.
  Uses a `DataRecord` (a Landlab xarray-backed particle tracker) to move sediment parcels
  along links.
- `FlowDirectorSteepest` -- works on `NetworkModelGrid` to set flow direction.

Most other Landlab components (overland flow, soil moisture, groundwater) are written
for `RasterModelGrid` and do **not** operate on `NetworkModelGrid`.

---

## 4. Is There a Grid Type for HRUs?

This is the key question for pywatershed compatibility, and the honest answer is:
**not directly**.

### Why VoronoiDelaunayGrid is close but not quite right

An HRU is an irregular polygon with known area, connectivity to neighboring HRUs, and
a defined flow path to downstream HRUs. `VoronoiDelaunayGrid` accepts arbitrary (x,y)
node positions and constructs Voronoi polygons as cells. If you place nodes at HRU
centroids, the resulting Voronoi cells approximate the HRU polygons -- but:

1. Landlab computes the Voronoi tessellation from the node positions. You cannot supply
   your own polygon boundaries. The cell areas will differ from true HRU areas.
2. The connectivity (which HRU drains to which) is inferred from the Delaunay
   triangulation, not from your watershed topology. Drainage network topology in a
   real watershed is determined by the DEM, not by proximity of HRU centroids.
3. `VoronoiDelaunayGrid` is 2D and assumes a planar domain. HRU networks are often
   best thought of as a directed graph, not a 2D spatial tessellation.

### Why NetworkModelGrid is also close but not quite right

`NetworkModelGrid` handles the directed graph aspect well -- you supply connectivity
explicitly -- but it has no cells and no areal extent. HRUs have area (for ET, soil
moisture, snowpack calculations), which is a cell quantity. A `NetworkModelGrid` node
can carry node-level fields but there is no built-in mechanism for area-weighted
accumulation.

### The most honest mapping

| pywatershed concept          | Best Landlab analog                             | Gap                                              |
| ---------------------------- | ----------------------------------------------- | ------------------------------------------------ |
| HRU (hillslope unit)         | Node on `NetworkModelGrid` (area as node field) | No cell geometry; area must be stored as a field |
| HRU polygon boundary         | Not representable                               | Voronoi approximation only                       |
| HRU-to-HRU flow connectivity | Links on `NetworkModelGrid`                     | Good match                                       |
| Channel reach                | Link on `NetworkModelGrid`                      | Good match                                       |
| Reach junction               | Node on `NetworkModelGrid`                      | Good match                                       |

The most pragmatic path: represent HRUs as **nodes** on a `NetworkModelGrid`, store HRU
area as a node field (`grid.at_node["drainage_area"]` or similar), and accept that
Landlab's cell-based geometry is simply unused. This loses Landlab's finite-volume
gradient/flux infrastructure but retains the shared-field namespace and component
interoperability.

---

## 5. Multi-Grid Models: Does Landlab Support Them?

**No -- not natively.**

All Landlab components are designed to operate on a single grid object. The grid is the
shared namespace; components access fields by string name through `grid.at_node[...]`,
`grid.at_link[...]`, etc. There is no built-in framework for:

- Running two components on different grid instances simultaneously
- Conservative remapping of fields between a raster hillslope grid and a network channel
  grid
- Any cross-grid flux aggregation (e.g., summing hillslope runoff into channel nodes)

### What does exist within a single grid

Landlab has a rich set of **intra-grid mappers** that move values between element types
on the same grid:

```python
# node -> link (mean, max, min, etc.)
grid.map_mean_of_link_nodes_to_link("my_node_field")
# link -> node
grid.map_max_of_links_to_node("my_link_field")
```

These are useful for staggered-grid finite-volume schemes but are not cross-grid mappings.

### How multi-grid coupling is done in practice

In Landlab-based research, hillslope-to-channel coupling is typically handled in one of
two ways:

1. **Single raster grid with flow accumulation**: Run hillslope processes on all nodes,
   use `FlowAccumulator` to route water/sediment downhill, read off discharge at channel
   nodes. Everything lives on one `RasterModelGrid`. The "channel" is implicitly defined
   by the flow network extracted from the DEM.

2. **Manual bridging**: Run a raster-based hillslope model and a `NetworkModelGrid`-based
   channel model separately, then write custom aggregation code to sum lateral inputs
   (runoff, sediment) from raster cells draining to each network link/node. Landlab
   provides the `channel_network_grid_tools` utility module and the
   `Create A Network Grid from Raster Grid` tutorial that set up the index mapping needed
   for this, but the aggregation itself is user code. There is no conservative-mapping
   operator built into Landlab.

This is a genuine gap in Landlab for pywatershed-style models, where the spatial domain
is inherently multi-scale (HRU hillslopes feeding a sparse channel network) and the two
representations are not simply coarsenings of each other.

---

## 6. Components Relevant to pywatershed Processes

The following Landlab components implement physics that overlaps with pywatershed's
process suite. All operate on `RasterModelGrid` unless noted.

### Soil moisture and infiltration

- `SoilMoistureDynamics` -- bucket-style soil moisture with ET loss and recharge.
  Node-based. Requires `vegetation__cover_fraction` and precipitation forcing.
- `InfiltrateGreenAmpt` (`infiltrate_soil_green_ampt`) -- Green-Ampt infiltration.

### Evapotranspiration

- `PotentialEvapotranspirationField` -- computes PET from radiation and temperature.
  Node-based.
- `Radiation` -- shortwave and net radiation at nodes.

### Overland flow

- `OverlandFlowBates` -- inertial shallow water (de Almeida/Bates scheme). Raster only.
- `ImplicitKinwaveOverlandFlow` -- implicit kinematic wave. Raster only.
- `LinearDiffusionOverlandFlowRouter` -- linearized diffusion wave. Raster only.
- `KinwaveOverlandFlowModel` -- explicit kinematic wave. Raster only.

### Subsurface / groundwater

- `GroundwaterDupuitPercolator` -- Dupuit-Boussinesq unconfined aquifer with seepage.
  Raster only. Demonstrated to conserve mass. Supports adaptive timestepping.

### Flow routing (prerequisite for most components)

- `FlowAccumulator` -- accumulates drainage area and discharge from runoff. Works on
  raster and `NetworkModelGrid`.
- `FlowDirectorSteepest`, `FlowDirectorD8`, `FlowDirectorMFD` -- flow direction
  algorithms. Raster; `FlowDirectorSteepest` also works on `NetworkModelGrid`.
- `PriorityFloodFlowRouter` -- combined pit-filling + flow direction + accumulation.

### Channel / network sediment

- `NetworkSedimentTransporter` -- Lagrangian bedload transport on `NetworkModelGrid`.
  Uses `DataRecord` for parcel tracking. Can ingest NHDPlus HR networks.

---

## 7. Compatibility Assessment: pywatershed + Landlab

### What aligns well

- **NetworkModelGrid topology** maps cleanly to pywatershed's HRU/reach graph if HRUs
  are treated as nodes and reaches as links.
- **Component physics** (soil moisture, ET, overland flow, groundwater) implements
  algorithms pywatershed needs, saving reimplementation effort.
- **BMI (Basic Model Interface)** -- Landlab components expose a BMI-compatible
  interface. pywatershed processes could in principle be wrapped as BMI clients and
  Landlab components as BMI servers, using CSDMS's standard coupling infrastructure.
  This would require no changes to either codebase and is the most defensible
  interoperability path.
- **Standard field names** -- Landlab uses CF-like standard names. If pywatershed
  Process variables are named to match, field handoff becomes trivial.

### What does not align

- **Grid-centricity vs. ref-passing**: Landlab components require a `ModelGrid` object
  and read/write fields via string keys on that grid. pywatershed processes share state
  via direct numpy buffer references with no grid object. Adapting a Landlab component
  as a pywatershed `Process` subclass requires wrapping the grid field access behind
  `get_inputs()` / `get_variables()` semantics -- doable but non-trivial.
- **No area-weighted HRU cells**: If HRUs are represented as nodes on a
  `NetworkModelGrid`, Landlab's cell-based gradient and flux divergence infrastructure
  is unavailable. Area-weighted ET, soil moisture, and runoff generation would need to
  be handled at the node level with explicit area fields.
- **No multi-grid conservative mapping**: Hillslope-to-channel aggregation must be
  written by hand. Landlab provides the index mapping tools but not the aggregation
  operator.
- **Most overland flow and soil moisture components are raster-only**: Directly reusing
  them on a `NetworkModelGrid` (HRU nodes) is not possible without modification or
  reimplementation.
- **Shared global state vs. explicit wiring**: Landlab's grid-as-namespace pattern
  means any component can read or overwrite any field. pywatershed's explicit wiring
  (declared inputs/outputs, reference sharing) provides stronger encapsulation. Mixing
  the two paradigms in a single model requires care to avoid one side silently
  overwriting the other's state.

### Recommended interoperability strategy

1. **Use BMI as the coupling layer.** Landlab ships `bmi_bridge`, which wraps Landlab
   components as BMI-compliant objects. pywatershed processes could be given a thin BMI
   wrapper. CSDMS's `babelflow` or manual BMI orchestration then couples them at a
   well-defined interface without requiring either side to know about the other's
   internal data structures.

2. **Define a shared HRU grid.** Create a `NetworkModelGrid` where nodes are HRUs and
   links are the drainage connectivity. Store HRU area, slope, and aspect as node
   fields. Run Landlab components that can operate on `NetworkModelGrid` directly
   (currently limited to flow routing). For raster-only components (overland flow, soil
   moisture), either (a) run them on a raster and aggregate outputs to network nodes, or
   (b) extract and generalize their physics into a pywatershed `Process` subclass.

3. **Target the NetworkSedimentTransporter first.** This is the most mature Landlab
   component on `NetworkModelGrid` and addresses channel sediment transport, which is
   not currently in pywatershed. It would be a clean addition via BMI coupling without
   requiring any grid-type compromises.

4. **Contribute a VoronoiDelaunayGrid variant that accepts user-supplied polygon
   boundaries.** This is a more ambitious path, but it would make Landlab genuinely
   HRU-native and open up the full raster-based component library to HRU-scale models.
   The gap is well-defined and Landlab's developer guide actively solicits contributions.

---

## 8. Open Questions

- **Can `VoronoiDelaunayGrid` be initialized with user-supplied Voronoi cells** (i.e.,
  actual HRU polygon boundaries derived from a GIS) rather than computing them from
  node positions? The documentation does not address this. A look at the source of
  `landlab/graph/voronoi/` would clarify.
- **Does `FramedVoronoiGrid` offer any traction here?** It starts from a rectangular
  grid and perturbs node positions randomly. Probably not useful for real HRU geometry,
  but worth confirming.
- **What does `channel_network_grid_tools` provide exactly?** The utility module is
  listed in the API but not well-documented. It may contain the raster-to-network
  index mapping needed for hillslope-to-channel aggregation.
- **Is there prior art in the Landlab community for HRU-based models?** The `DupuitPercolator`
  groundwater tutorial uses a raster with a conceptual catchment framing that hints at
  HRU thinking. Worth searching the Landlab issue tracker and publications list.

---

_Sources: Landlab readthedocs (Introduction to Landlab's Gridding Library,
NetworkSedimentTransporter tutorial, Groundwater tutorial), Landlab API reference
sidebar, pywatershed `incarnations/xr/base.py` and `design_summary.md`._

---

## 9. Architecture Discussion: pywatershed Discretization and BMI Coupling

_The following summarizes a design discussion about how pywatershed's spatial concepts
should be formalized and how Landlab coupling should be structured._

### The pywatershed Discretization concept

pywatershed already has an informal notion of a **discretization**: a set of spatial
parameters (HRU areas, connectivity, slopes, lengths, etc.) that are shared by all
processes operating on the same spatial domain, and are distinct from process-specific
parameters. This is analogous to Landlab's `ModelGrid` in intent, though implemented
very differently (currently as shared entries in a parameter `Dataset` rather than a
first-class object).

### Formalizing Discretization as a first-class object

Making `Discretization` a formal class -- wrapping an `xr.Dataset` with spatial
metadata and its own accessors/methods -- is a worthwhile investment for two reasons:

1. **MPI domain decomposition.** When MPI parallelism is introduced, the discretization
   is the natural unit of partitioning. It defines which HRUs live on which rank and
   which are halo/ghost cells for inter-rank communication. Defining this as a
   first-class object now avoids a painful retrofit once the MPI topology is in place.

2. **BMI coupling surface.** A well-defined `Discretization` object provides a clean
   basis for what pywatershed exposes through BMI `get_value` / `set_value` calls --
   the spatial granularity and field layout are unambiguous.

The `Discretization` object should own not just the spatial attributes but also the
MPI partition map and, eventually, references to the pre-computed conservative mapping
operators described below.

### Recommended coupling architecture: separate grids, offline mapping, BMI exchange

The cleanest path to Landlab interoperability does not require pywatershed HRUs to be
represented as a Landlab grid type. Instead:

1. **pywatershed maintains its HRU-scale `Discretization`** -- a directed graph of HRUs
   with known areas, connectivity, slopes, and other spatial attributes.

2. **Landlab maintains a separate, higher-resolution grid** appropriate to the physics
   being computed (e.g., a `RasterModelGrid` derived from a DEM for overland flow or
   groundwater, or a `NetworkModelGrid` for channel sediment transport).

3. **Conservative mapping operators are computed offline** (from GIS/DEM analysis,
   prior to any model run) and stored as sparse weight matrices alongside the
   discretization -- for example in a NetCDF file. These operators map between HRU-scale
   quantities and the Landlab grid's resolution in a mass-conservative way.

4. **BMI is the runtime coupling layer.** pywatershed advances its processes, exposes
   HRU-scale outputs (runoff, ET demand, soil moisture, etc.) via BMI `get_value`, the
   coupling layer applies the conservative mapping, Landlab receives the remapped fields,
   runs its component, and returns results via BMI `set_value`. Neither model needs to
   know about the other's internal grid structure.

This approach:

- Keeps the two models cleanly separated and independently testable
- Makes the spatial mapping an auditable, version-controlled artifact
- Avoids the topology problem of trying to force watershed drainage connectivity into a
  Landlab grid type whose connectivity is inferred from geometry
- Is compatible with MPI parallelism, since the mapping operators can be partitioned
  alongside the discretization

### Near-term proof of concept

`NetworkSedimentTransporter` on a `NetworkModelGrid` seeded from the pywatershed HRU
connectivity is the most natural first target. Channel sediment transport is not
currently in pywatershed, the `NetworkModelGrid` topology is explicitly user-supplied
(so the drainage network is correct by construction), and the BMI coupling surface is
well-defined: pywatershed provides discharge per reach, Landlab transports sediment and
returns bed state.
