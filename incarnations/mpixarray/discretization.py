"""
discretization.py
=================
The Discretization: the unit of spatial decomposition and the owner of a
grid -- its decomposition, its own (dis) parameters, and derived
topology. Foundational module -- it imports no other
incarnations/mpixarray module; Model/ModelMPI import ``Discretization``
from here.

Scope:
  - SPACE decomposition. serial: a degenerate placeholder (``comm=None``);
    ``decompose`` is identity. MPI: wraps mpixarray's ``parallelize`` over
    ``dims`` and carries the comm.
  - DIS parameters (July 2026): grid-owned variables (areas, unit
    conversions, topology like ``tosegment``) -- pywatershed's
    dis_hru/dis_seg split (utils/separate_nhm_params.py). Passed at
    construction (``parameters=``); the Model seeds each grid's shared
    dataset from here FIRST (dis is the first-priority parameter source),
    so processes read dis variables uniformly via ``self._obj[name]``.
  - Derived topology (July 2026): ``topological_order`` -- a GENERIC
    to-index ordering (Discretization stays the uniform, no-subclass
    object), replicating pywatershed PRMSChannel's networkx construction
    EXACTLY. The order is not unique, and a different (equally valid)
    order changes floating-point accumulation downstream -- hence
    replicate, don't improve (a dependency-free ordering is a possible
    later step, with a tolerance decision attached).

Time streaming (``set_streaming``) stays with the Model/Time loop and runs
on the decomposed dataset.

The MPI path reaches ``parallelize`` through the ``ds.mpi`` accessor, which
ModelMPI registers by importing mpixarray; this module therefore needs no
mpixarray import of its own (and the serial path, ``comm=None``, never
calls it). networkx is imported lazily, only when ``topological_order``
is called.
"""

import pathlib as pl
from typing import Any

import numpy as np
import xarray as xr


class Discretization:
    """Unit of spatial decomposition; a grid.

    Identity is the Model's ``discretizations`` dict key, not stored here.
    serial: degenerate (``comm=None``, full extent, ``decompose`` is identity).
    MPI: wraps ``parallelize`` over ``dims`` and carries the comm.

    ``parameters`` (optional) is the grid-OWNED dataset (dis_hru/dis_seg
    style: areas, conversions, topology); the Model treats it as the
    first-priority source for processes' declared parameters, and derived
    topology methods (``topological_order``) read from it directly.
    """

    def __init__(
        self,
        dims: list[str],
        *,
        comm: Any = None,
        parameters: xr.Dataset | pl.Path | None = None,
        topo_order: dict[str, str] | None = None,
        topo_one_based: bool = True,
    ) -> None:
        self.dims = list(dims)
        self.comm = comm
        if isinstance(parameters, pl.Path):
            parameters = xr.open_dataset(parameters)
        self.parameters = parameters
        self.dataset: Any = (
            None  # the grid's shared dataset (set by the Model)
        )
        self._topo_order_cache: dict[tuple[str, bool], np.ndarray] = {}
        # topo_order: {new_var: to_index_var} (the Map dict idiom) --
        # compute the ordering at construction and store it as a DIS
        # parameter, so a process receives it by DECLARATION (dis-first
        # sourcing) like any other dis variable; no dis-object access
        # path is needed inside Process. topo_one_based: see
        # topological_order (applies to every topo_order entry).
        if topo_order is not None:
            for new_var, to_index in topo_order.items():
                if self.parameters is None:
                    raise ValueError(
                        "topo_order requires 'parameters' (the to-index "
                        f"variable '{to_index}' must live there)."
                    )
                self.parameters[new_var] = (
                    self.parameters[to_index].dims,
                    self.topological_order(to_index, one_based=topo_one_based),
                )

    @property
    def is_distributed(self) -> bool:
        return self.comm is not None

    def decompose(self, ds: Any) -> Any:
        """Return the space-decomposed dataset on this grid.

        serial (``comm is None``): identity. MPI: ``parallelize`` over
        ``dims``, updating ``self.comm`` to the comm ``parallelize`` returns.
        """
        if self.comm is None:
            return ds
        ds_mpi, self.comm = ds.mpi.parallelize(
            dims=self.dims, scheme="single", comm=self.comm
        )
        return ds_mpi

    def topological_order(
        self, to_index: str = "tosegment", one_based: bool = True
    ) -> np.ndarray:
        """Upstream-to-downstream ordering from a to-index variable.

        ``to_index`` names a connectivity variable on this dis's
        ``parameters``. ``one_based=True`` (default) is the PRMS-legacy
        convention (``tosegment`` from legacy files: 1-based, 0 =
        outlet); ``one_based=False`` is the native FlowGraph convention
        (``to_graph_index``: 0-based, -1 = outlet). Returns 0-based
        indices ordered so every element precedes the element it flows
        to. Cached per (variable, convention).

        Replicates pywatershed PRMSChannel._initialize_channel_data
        EXACTLY (networkx DiGraph + topological_sort, isolated outlets
        prepended): the order is not unique and downstream accumulation
        (`+=` over upstream neighbors) is float-order-sensitive, so
        matching pywatershed answers at 1e-13 requires the SAME order.
        """
        cache_key = (to_index, one_based)
        if cache_key in self._topo_order_cache:
            return self._topo_order_cache[cache_key]
        if self.parameters is None or to_index not in self.parameters:
            raise ValueError(
                f"topological_order: '{to_index}' is not a variable of "
                "this discretization's parameters."
            )
        import networkx as nx  # lazy: only topology users need it

        # normalize to 0-based; negative = outlet
        if one_based:
            to_seg = (self.parameters[to_index].values - 1).astype("int64")
        else:
            to_seg = self.parameters[to_index].values.astype("int64")
        n_seg = to_seg.shape[0]
        outflow_mask = np.full((n_seg,), False)
        connectivity = []
        for iseg in range(n_seg):
            if to_seg[iseg] < 0:
                outflow_mask[iseg] = True
                continue
            connectivity.append((iseg, to_seg[iseg]))

        if n_seg > 1:
            graph = nx.DiGraph()
            graph.add_edges_from(connectivity)
            order = list(nx.topological_sort(graph))
        else:
            order = [0]

        # Isolated segments (no upstream AND no downstream) never enter
        # the graph; throw them back at the top of the order (pywatershed
        # does exactly this).
        wh_mask_set = set(np.where(outflow_mask)[0])
        mask_not_in_order = list(wh_mask_set - set(order))
        if len(mask_not_in_order):
            order = mask_not_in_order + order

        order_arr = np.array(order, dtype="int64")
        self._topo_order_cache[cache_key] = order_arr
        return order_arr
