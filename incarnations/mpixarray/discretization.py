"""
discretization.py
=================
The Discretization: the unit of spatial decomposition and (eventually) the owner
of a grid. Foundational module -- it imports no other incarnations/mpixarray
module; Model/ModelMPI import ``Discretization`` from here.

Spike scope: owns the SPACE decomposition only.
  - serial: a degenerate placeholder (``comm=None``); ``decompose`` is identity.
  - MPI: wraps mpixarray's ``parallelize`` over ``dims`` and carries the comm.

Time streaming (``set_streaming``) stays with the Model/Time loop and runs on the
decomposed dataset. Grid metadata/topology, dataset-ownership, and multiple
discretizations + Maps are future work (see pws_phoenix/CLAUDE.md).

The MPI path reaches ``parallelize`` through the ``ds.mpi`` accessor, which
ModelMPI registers by importing mpixarray; this module therefore needs no
mpixarray import of its own (and the serial path, ``comm=None``, never calls it).
"""

from typing import Any


class Discretization:
    """Unit of spatial decomposition; a grid, identified by ``name``.

    ``dims`` (the spatial dim(s)) defaults to ``[name]`` -- the common case
    where the grid is a single dimension named like itself.
    serial: degenerate (``comm=None``, full extent, ``decompose`` is identity).
    MPI: wraps ``parallelize`` over ``dims`` and carries the comm.
    """

    def __init__(
        self, name: str, dims: list[str] | None = None, *, comm: Any = None
    ) -> None:
        self.name = name
        self.dims = list(dims) if dims is not None else [name]
        self.comm = comm

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
