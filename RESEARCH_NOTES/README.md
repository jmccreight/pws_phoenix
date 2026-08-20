# RESEARCH_NOTES

## Contents

| File | What it is |
|---|---|
| `river-routing-dag-vs-parallel.md` | **The substantive note.** Is the DAG/topological solve in Muskingum-style stream routing necessary, or can routing be made embarrassingly parallel by lagging upstream inflow one substep? Numerics, the parameter coupling that blocks naive dt reduction, literature, and a proposed experiment sequence. Start here. |
| `notes-index.md` | A Kiro steering index. Only functional inside a `.kiro/steering/` directory. Inert here. |
| `routing-parallelism-note.md` | A breadcrumb originally written into the pywatershed repo. Describes pywatershed file paths and points back at the canonical note. Its code-map content is already folded into `river-routing-dag-vs-parallel.md`, so it is redundant here. |

## Provenance

These were produced in a conversation with the `pywatershed` repo open
(`/Users/jmccreight/usgs/pywatershed`), then copied here because this is the
project the question actually belongs to. Originals:

```
/Users/jmccreight/.kiro/steering/river-routing-dag-vs-parallel.md
/Users/jmccreight/.kiro/steering/notes-index.md
/Users/jmccreight/usgs/pywatershed/.kiro/steering/routing-parallelism-note.md
```

Copies here were byte-identical to those originals (verified by MD5). If you
edit here, the copies diverge -- decide which location is canonical and prune
the other, or the next agent to read both will get conflicting versions.

## Caveats before you build on this

**The `inclusion:` YAML front matter is inert in this folder.** It is Kiro
steering metadata and only takes effect inside a `.kiro/steering/` directory. If
you want the note automatically available to an agent working in this project,
it needs to live at `pws_phoenix/.kiro/steering/` rather than (or in addition
to) here. `notes-index.md` in particular has no purpose outside such a
directory.

**Nothing in these notes has been verified experimentally.** No code was
written, no experiments run. Every performance and error statement is a
prediction.

**Literature claims came from search snippets and abstracts only.** Full-text
fetches returned HTTP 403 for the Wiley, AMS, MDPI and eScholarship items.
Titles, DOIs and URLs are reliable; the summaries of what each paper
*concludes* should be checked against the actual papers before being cited or
built upon. The most valuable missing number is the quantitative result in
David et al. (2013, doi:10.1002/wrcr.20250): how many reaches upstream the
within-timestep influence remains significant. That figure bears directly on
whether a k-hop or halo scheme is viable, and it was not retrieved.

**Code claims are reliable.** File structure, line numbers, the `c0` clamp
behaviour and the FlowGraph API were read directly from the pywatershed source.
The algebra (Muskingum coefficient formulas, the positivity window
`2*K*x <= dt <= 2*K*(1-x)`, the first-order error term) is derivable from that
source and was checked by hand.

## Suggested first step

Experiment 1 in the note: diagnostics only, no new physics. Distributions of K,
`ts`/`tsi`, and `c0`/`c1`/`c2` on real domains, plus the topological level
profile (depth and width per level). It is cheap, it replaces the note's
back-of-envelope scaling estimates with measurements, and it determines how much
exact parallelism is available before any accuracy is traded away.

Also flagged in the note: pywatershed already contains
`pywatershed/utils/mmr_to_mf6_dfw.py`, `examples/07_mmr_to_mf6_chf_dfw.ipynb`
and `autotest/test_mmr_to_mf6_dfw.py` -- an MMR to MODFLOW 6 CHF/DFW
(diffusive wave) conversion with a test harness. That supplies a
higher-fidelity reference solution, so routing experiments can be scored
against the diffusive wave rather than only against the legacy DAG-Muskingum
answer. Worth reading before designing anything.
