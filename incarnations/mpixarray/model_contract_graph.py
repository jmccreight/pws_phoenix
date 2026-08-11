"""ModelContractGraph: the input contract, drawn.

The successor to pywatershed's ModelGraph, extended for the
pws_phoenix object model: processes live on GRIDS (rendered as
clusters), Maps carry variables between grids (cross-cluster edges),
external inputs enter from outside, and the Time global feeds every
process. Built purely from the declarations via ``Model.input_spec``
-- no data are loaded and no model is constructed.

Backend-agnostic by design: the constructor builds a small
intermediate representation -- ``grids`` (cluster -> process nodes),
``externals`` and typed ``edges`` -- with two renderers beside it:

- ``to_dot()`` (graphviz source): the STRUCTURED renderer -- data
  flow runs along one axis (rankdir=LR), with prior-step back-edges
  excluded from the ranking. Preferred when the graphviz package/
  binary is available; the method itself needs neither.
- ``to_mermaid()``: the zero-dependency fallback -- renders in
  JupyterLab (>= 4.1) markdown and on GitHub, but its layout
  scatters on dense configurations (mermaid offers no rank control).

Edge semantics:

- solid, labeled: INTERNAL inputs -- a variable (or internal
  parameter, or same-named supplied parameter) produced on the grid
  and consumed by structural sharing. One edge per
  producer->consumer pair; the label lists the variables carried.
- dashed, labeled: MAP-FED inputs -- the hru->segment (etc.)
  couplings; the label is the mapped variable (renames shown as
  ``source>target``).
- from a stadium node: EXTERNAL inputs -- what must be supplied.
- dotted, from Time: the model clock reaches every process (drawn
  once per grid cluster to keep the figure readable).

The SUPPLY side of the contract is drawn in gray, INSIDE each
process node (an HTML-like table): a white header (the process --
computed) over gray sections for the required parameters (authored
static | time-varying, classified from the declared dims; DERIVABLE
-- required but obtainable by their declared derivation -- as their
own section), the ``initial=`` value seams, and the RESTARTABLE
INITIAL STATE (every ``restart=True`` variable is a settable
initial condition -- a warm start supplies all of them). Section
headers always carry counts; ``show_params=True`` expands the
names. External inputs are gray nodes. Together with the edges,
that IS the contract: gray = you supply it, white = the model
computes it.

Usage (e.g. in a notebook):

    graph = ModelContractGraph(process_dict, maps=maps)
    print(graph.to_mermaid())     # or:
    graph                         # renders in JupyterLab >= 4.1
"""

from typing import Any

from model import Model
from process import _dict_of_kind

# how many variable names an edge label lists before eliding
_LABEL_MAX = 3

# a parameter with one of these dims is TIME-VARYING (cyclic --
# indexed by a derived calendar coordinate, not model time)
_CYCLIC_DIMS = ("nmonth", "ndoy")


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
        spec = Model.input_spec(process_dict, maps=maps, include_optional=True)
        self.show_params = show_params
        # schedule position (process_dict order): edges pointing
        # AGAINST it are prior-step back-edges by construction and
        # must not drive the flow-axis layout
        self._order = {slot: ii for ii, slot in enumerate(process_dict)}

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

        # -- the supply side of the contract, per process: required
        # parameters (authored static vs time-varying, classified
        # from the declared dims; DERIVABLE ones -- carrying a
        # declared derivation -- are their own class) and the
        # initial-value seams --
        self.parameters: dict[str, dict[str, list[str]]] = {}
        for slot, entry in process_dict.items():
            static: list[str] = []
            cyclic: list[str] = []
            derivable: list[str] = []
            for name, meta in _dict_of_kind(
                entry["class"], "parameter"
            ).items():
                if meta.derivation is not None:
                    derivable.append(name)
                elif any(dd in _CYCLIC_DIMS for dd in meta.dims):
                    cyclic.append(name)
                else:
                    static.append(name)
            self.parameters[slot] = {
                "static": static,
                "cyclic": cyclic,
                "derivable": derivable,
            }
        self.initial_values: dict[str, list[str]] = {}
        for grid, gg in spec["required"].items():
            for init_name, info in gg["initial_values"].items():
                self.initial_values.setdefault(info["process"], []).append(
                    init_name
                )
        # every restart=True variable is a settable initial condition
        # (a warm start supplies all of them; see Model.write_restart)
        self.restart_vars: dict[str, list[str]] = {
            slot: list(entry["class"].get_restart_variables())
            for slot, entry in process_dict.items()
        }

        # the Maps' supply requirement (spec["maps"]): one weights
        # matrix per map, summarized per (source, target) grid pair
        self.map_weights: dict[tuple[str, str], dict[str, Any]] = {}
        for map_name, info in spec.get("maps", {}).items():
            key = (info["source_grid"], info["target_grid"])
            entry = self.map_weights.setdefault(
                key,
                {
                    "count": 0,
                    "derived": 0,
                    "shape": info["weights_shape"],
                },
            )
            entry["count"] += 1
            if info["derivation"]:
                entry["derived"] += 1

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
                    internal.setdefault((producer, consumer), []).append(name)
        self.internal_edges: list[tuple[str, str, list[str]]] = [
            (src, dst, names) for (src, dst), names in internal.items()
        ]

        # -- map edges: source-grid producer -> consumer, per map --
        # the producer is whichever source-grid process declares the
        # source variable (as a variable, internal parameter, or input
        # -- e.g. an external-forcing carrier)
        def _producer_of(grid: str, var: str) -> str:
            for slot, entry in process_dict.items():
                if proc_grid[slot] != grid:
                    continue
                cls = entry["class"]
                if var in cls.get_var_names() or var in tuple(
                    cls.get_parameters_internal()
                ) + tuple(cls.get_inputs()):
                    return slot
            return f"({grid})"

        # aggregated per (producer, consumer) pair, like the internal
        # edges -- separate arcs per variable dominate the drawing
        map_pairs: dict[tuple[str, str], list[str]] = {}
        for grid, gg in spec["optional"].items():
            for name, info in gg["map_fed_inputs"].items():
                label = (
                    name
                    if info["source_var"] == name
                    else f"{info['source_var']}>{name}"
                )
                src = _producer_of(info["source_grid"], info["source_var"])
                for consumer in info["consumers"]:
                    map_pairs.setdefault((src, consumer), []).append(label)
        self.map_edges: list[tuple[str, str, list[str]]] = [
            (src, dst, labels) for (src, dst), labels in map_pairs.items()
        ]

    # -- renderers ------------------------------------------------------

    def _param_counts(self, slot: str) -> str:
        """The bubble's one-line parameter summary ('' if none)."""
        pp = self.parameters[slot]
        if not any(pp.values()):
            return ""
        counts = f"params: {len(pp['static'])} static"
        if pp["cyclic"]:
            counts += f" + {len(pp['cyclic'])} tv"
        if pp["derivable"]:
            counts += f" + {len(pp['derivable'])} derivable"
        return counts

    def _dot_table(self, slot: str, cls_name: str) -> str:
        """The process node as an HTML-like table: a white header (the
        process -- computed) over GRAY sections for everything the
        modeler supplies -- parameters (static | time-varying),
        initial values, and the restartable initial state. Section
        headers always show counts; ``show_params=True`` expands the
        names."""

        def section(title: str, names: list[str]) -> str:
            body = f"<B>{title}</B>"
            if self.show_params and names:
                for nn in sorted(names):
                    body += f'<BR ALIGN="LEFT"/>{nn}'
                body += '<BR ALIGN="LEFT"/>'
            return (
                '<TR><TD BGCOLOR="gray90" ALIGN="LEFT">'
                f'<FONT POINT-SIZE="9">{body}</FONT></TD></TR>'
            )

        pp = self.parameters[slot]
        rows = [
            f'<TR><TD BGCOLOR="white"><B>{slot}</B><BR/>{cls_name}</TD></TR>'
        ]
        if pp["static"]:
            rows.append(
                section(
                    f"parameters: {len(pp['static'])} static", pp["static"]
                )
            )
        if pp["cyclic"]:
            rows.append(
                section(
                    f"parameters: {len(pp['cyclic'])} time-varying",
                    pp["cyclic"],
                )
            )
        if pp["derivable"]:
            rows.append(
                section(
                    f"parameters: {len(pp['derivable'])} derivable",
                    pp["derivable"],
                )
            )
        inits = self.initial_values.get(slot, [])
        if inits:
            rows.append(section(f"initial values: {len(inits)}", inits))
        state = self.restart_vars.get(slot, [])
        if state:
            rows.append(
                section(f"initial state (restartable): {len(state)}", state)
            )
        return (
            '<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" '
            'CELLPADDING="4">' + "".join(rows) + "</TABLE>"
        )

    @staticmethod
    def _label(names: list[str]) -> str:
        if len(names) <= _LABEL_MAX:
            return ", ".join(names)
        shown = ", ".join(names[:_LABEL_MAX])
        return f"{shown} +{len(names) - _LABEL_MAX}"

    def to_mermaid(self, direction: str = "TB") -> str:
        """Render the graph as mermaid flowchart text (`direction`:
        "TB" portrait or "LR" landscape; mermaid decides the rest of
        the layout itself)."""
        out = [f"flowchart {direction}"]
        out.append('    time(["Time (the model clock)"])')
        for grid, procs in self.grids.items():
            out.append(f'    subgraph {grid}["grid: {grid}"]')
            for slot, cls_name, _n_params in procs:
                label = f"{slot}<br/>{cls_name}"
                counts = self._param_counts(slot)
                if counts:
                    label += f"<br/>{counts}"
                out.append(f'        {slot}["{label}"]')
            out.append("    end")
            out.append(f"    time -.-> {grid}")
        for grid, ext in self.externals.items():
            for name, consumers in ext.items():
                out.append(f'    ext_{name}(["{name}"])')
                for consumer in consumers:
                    out.append(f"    ext_{name} --> {consumer}")
        for slot, inits in self.initial_values.items():
            for init_name in inits:
                out.append(
                    f'    init_{init_name}(["initial value: {init_name}"])'
                )
                out.append(f"    init_{init_name} --> {slot}")
        for (src_grid, tgt_grid), ww in self.map_weights.items():
            label = (
                f"Map weights: {ww['count']} "
                f"({ww['shape'][0]} x {ww['shape'][1]})"
            )
            node = f"weights_{src_grid}_{tgt_grid}"
            out.append(f'    {node}(["{label}"])')
            out.append(f"    {node} -.- {tgt_grid}")
        for src, dst, names in self.internal_edges:
            out.append(f'    {src} -- "{self._label(names)}" --> {dst}')
        for src, dst, labels in self.map_edges:
            out.append(f'    {src} -. "{self._label(labels)}" .-> {dst}')
        return "\n".join(out)

    def to_markdown(self) -> str:
        """The mermaid text in a fenced block (renders on GitHub and
        in JupyterLab >= 4.1 markdown output)."""
        return f"```mermaid\n{self.to_mermaid()}\n```"

    def _repr_markdown_(self) -> str:
        return self.to_markdown()

    def to_dot(
        self,
        rankdir: str = "LR",
        size: float | str | None = None,
    ) -> str:
        """Render the graph as graphviz dot source -- the STRUCTURED
        renderer: data flow runs along one axis. The schedule order
        drives the ranking; prior-step back-edges (snow -> canopy
        etc.) are drawn but excluded from it (``constraint=false``),
        which is what keeps the axis clean -- mermaid has no such
        control. Render with the ``graphviz`` package/binary
        (``graphviz.Source(graph.to_dot())``) or any dot viewer; this
        method itself needs neither.

        Args:
            rankdir: the flow axis -- ``"LR"`` (landscape, default)
                or ``"TB"`` (portrait; fits page/notebook widths).
            size: maximum drawing size in INCHES -- a number caps
                both dimensions (e.g. ``10``), a string is passed to
                graphviz verbatim (e.g. ``"8,11"`` for width,height;
                append ``!`` to also scale up). The drawing scales
                DOWN proportionally to fit. None (default) = natural
                size.
        """
        out = [
            "digraph model_contract {",
            f"    rankdir={rankdir};",
            "    compound=true;",
        ]
        if size is not None:
            out.append(f'    size="{size}";')
        out += [
            '    fontname="Helvetica";',
            '    node [shape=box, style=rounded, fontname="Helvetica"];',
            '    edge [fontname="Helvetica", fontsize=10];',
            '    time [label="Time\\n(the model clock)", shape=oval,'
            " style=dashed];",
        ]
        for grid, procs in self.grids.items():
            out.append(f"    subgraph cluster_{grid} {{")
            out.append(f'        label="grid: {grid}";')
            out.append(
                '        style="rounded,filled"; fillcolor=gray98;'
                " color=gray60;"
            )
            for slot, cls_name, _n_params in procs:
                out.append(
                    f"        {slot} [shape=plain, "
                    f"label=<{self._dot_table(slot, cls_name)}>];"
                )
            out.append("    }")
            first = self.grids[grid][0][0]
            out.append(
                f"    time -> {first} [lhead=cluster_{grid}, "
                "style=dotted, arrowhead=none, constraint=false];"
            )
        for grid, ext in self.externals.items():
            for name, consumers in ext.items():
                out.append(
                    f'    ext_{name} [label="{name}", shape=ellipse, '
                    "style=filled, fillcolor=gray90, "
                    'fontname="Helvetica-Bold"];'
                )
                for consumer in consumers:
                    out.append(f"    ext_{name} -> {consumer};")
        # the Maps' weights requirement: one gray supply note per grid
        # pair, tied to the target cluster
        for (src_grid, tgt_grid), ww in self.map_weights.items():
            shape = f"({ww['shape'][0]} x {ww['shape'][1]})"
            label = (
                f"Map weights: {ww['count']} "
                f"{'matrices' if ww['count'] > 1 else 'matrix'} "
                f"{shape}"
            )
            if ww["derived"]:
                label += f"\\n{ww['derived']} with a known derivation"
            node = f"weights_{src_grid}_{tgt_grid}"
            first = self.grids[tgt_grid][0][0]
            out.append(
                f'    {node} [label="{label}", shape=note, '
                "style=filled, fillcolor=gray90, fontsize=9, "
                'fontname="Helvetica-Bold"];'
            )
            out.append(
                f"    {node} -> {first} [lhead=cluster_{tgt_grid}, "
                "style=dotted, arrowhead=none, constraint=false];"
            )
        big = len(self._order)
        for src, dst, names in self.internal_edges:
            attrs = [f'label="{self._label(names)}"']
            if self._order.get(src, big) > self._order.get(dst, big):
                attrs.append("constraint=false")  # prior-step back-edge
            out.append(f"    {src} -> {dst} [{', '.join(attrs)}];")
        for src, dst, labels in self.map_edges:
            out.append(
                f"    {src} -> {dst} "
                f'[label="{self._label(labels)}", style=dashed];'
            )
        out.append("}")
        return "\n".join(out)
