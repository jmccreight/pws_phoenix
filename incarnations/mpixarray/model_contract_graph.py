"""ModelContractGraph: the input contract, drawn.

The successor to pywatershed's ModelGraph, extended for the
pws_phoenix object model: processes live on GRIDS (rendered as
clusters), Maps carry variables between grids (cross-cluster edges),
external inputs enter from outside, and the Time global feeds every
process. Built purely from the declarations via ``Model.input_spec``
-- no data are loaded and no model is constructed.

Backend-agnostic by design (mermaid may prove insufficient for dense
configurations): the constructor builds a small intermediate
representation -- ``grids`` (cluster -> process nodes), ``externals``
and typed ``edges`` -- and ``to_mermaid()`` renders it; further
renderers (e.g. graphviz) can be added beside it without touching the
builder.

Edge semantics:

- solid, labeled: INTERNAL inputs -- a variable (or derived
  parameter, or same-named supplied parameter) produced on the grid
  and consumed by structural sharing. One edge per
  producer->consumer pair; the label lists the variables carried.
- dashed, labeled: MAP-FED inputs -- the hru->segment (etc.)
  couplings; the label is the mapped variable (renames shown as
  ``source>target``).
- from a stadium node: EXTERNAL inputs -- what must be supplied.
- dotted, from Time: the model clock reaches every process (drawn
  once per grid cluster to keep the figure readable).

Usage (e.g. in a notebook):

    graph = ModelContractGraph(process_dict, maps=maps)
    print(graph.to_mermaid())     # or:
    graph                         # renders in JupyterLab >= 4.1
"""

from typing import Any

from model import Model

# how many variable names an edge label lists before eliding
_LABEL_MAX = 3


class ModelContractGraph:
    """Build the contract graph for a configuration (same inputs as
    ``Model.input_spec``: only each entry's class + grid, and the
    maps' wiring, are read)."""

    def __init__(
        self,
        process_dict: dict[str, dict[str, Any]],
        maps: dict[str, Any] | None = None,
        show_params: bool = False,
    ) -> None:
        maps = maps or {}
        spec = Model.input_spec(
            process_dict, maps=maps, include_optional=True
        )
        self.show_params = show_params

        # -- clusters: grid -> [(slot, class name, n params)] --
        self.grids: dict[str, list[tuple[str, str, int]]] = {}
        proc_grid: dict[str, str] = {}
        for slot, entry in process_dict.items():
            grid = Model._resolve_grid(entry)
            proc_grid[slot] = grid
            self.grids.setdefault(grid, []).append(
                (
                    slot,
                    entry["class"].__name__,
                    len(entry["class"].get_parameters()),
                )
            )

        # -- external inputs: grid -> {name: [consumers]} --
        self.externals: dict[str, dict[str, list[str]]] = {}
        for grid, gg in spec["required"].items():
            self.externals[grid] = {
                name: info["consumers"]
                for name, info in gg["external_inputs"].items()
            }

        # -- internal edges, aggregated per (producer, consumer) --
        internal: dict[tuple[str, str], list[str]] = {}
        for grid, gg in spec["optional"].items():
            for name, info in gg["internal_inputs"].items():
                producer = info["producer"]
                if producer.startswith("("):
                    # satisfied by a same-named supplied parameter --
                    # a supply fact, not a process-to-process flow
                    continue
                for consumer in info["consumers"]:
                    internal.setdefault((producer, consumer), []).append(
                        name
                    )
        self.internal_edges: list[tuple[str, str, list[str]]] = [
            (src, dst, names) for (src, dst), names in internal.items()
        ]

        # -- map edges: source-grid producer -> consumer, per map --
        # the producer is whichever source-grid process declares the
        # source variable (as a variable, derived parameter, or input
        # -- e.g. an external-forcing carrier)
        def _producer_of(grid: str, var: str) -> str:
            for slot, entry in process_dict.items():
                if proc_grid[slot] != grid:
                    continue
                cls = entry["class"]
                if var in cls.get_var_names() or var in tuple(
                    cls.get_parameters_derived()
                ) + tuple(cls.get_inputs()):
                    return slot
            return f"({grid})"

        self.map_edges: list[tuple[str, str, str]] = []
        for grid, gg in spec["optional"].items():
            for name, info in gg["map_fed_inputs"].items():
                label = (
                    name
                    if info["source_var"] == name
                    else f"{info['source_var']}>{name}"
                )
                src = _producer_of(info["source_grid"], info["source_var"])
                for consumer in info["consumers"]:
                    self.map_edges.append((src, consumer, label))

    # -- renderers ------------------------------------------------------

    @staticmethod
    def _label(names: list[str]) -> str:
        if len(names) <= _LABEL_MAX:
            return ", ".join(names)
        shown = ", ".join(names[:_LABEL_MAX])
        return f"{shown} +{len(names) - _LABEL_MAX}"

    def to_mermaid(self) -> str:
        """Render the graph as mermaid flowchart text."""
        out = ["flowchart TB"]
        out.append('    time(["Time (the model clock)"])')
        for grid, procs in self.grids.items():
            out.append(f'    subgraph {grid}["grid: {grid}"]')
            for slot, cls_name, n_params in procs:
                label = f"{slot}<br/>{cls_name}"
                if self.show_params:
                    label += f"<br/>({n_params} parameters)"
                out.append(f'        {slot}["{label}"]')
            out.append("    end")
            out.append(f"    time -.-> {grid}")
        for grid, ext in self.externals.items():
            for name, consumers in ext.items():
                out.append(f'    ext_{name}(["{name}"])')
                for consumer in consumers:
                    out.append(f"    ext_{name} --> {consumer}")
        for src, dst, names in self.internal_edges:
            out.append(f'    {src} -- "{self._label(names)}" --> {dst}')
        for src, dst, label in self.map_edges:
            out.append(f'    {src} -. "{label}" .-> {dst}')
        return "\n".join(out)

    def to_markdown(self) -> str:
        """The mermaid text in a fenced block (renders on GitHub and
        in JupyterLab >= 4.1 markdown output)."""
        return f"```mermaid\n{self.to_mermaid()}\n```"

    def _repr_markdown_(self) -> str:
        return self.to_markdown()
