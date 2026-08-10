---
name: notebooks
description: Write user-facing example notebooks as py:percent .py files (the required mechanism) — format mechanics, the standard open-me header, and the pywatershed-numbered-notebook tone.
---

# Writing example notebooks (py:percent)

Applies to anything under an `examples/` directory in this repo.
Notebooks are **py:percent `.py` files, never committed `.ipynb`** —
they are git-diffable, ruff/mypy-checkable, runnable as plain scripts
(which is how they are tested), and open as real notebooks via
jupytext.

## Numbering and data directories

- Notebooks are NUMBERED to provide reading order: `XX_name.py`
  (`00_input_contract.py`, `01_prms_legacy_translation.py`, ...),
  concept-first. Cross-reference by the numbered filename.
- A notebook that writes anything writes ONLY under its own data
  directory `XX_name/` beside itself (create it in the notebook;
  gitignored via the `examples/[0-9][0-9]_*/` pattern). One notebook,
  one directory — cleanup and .gitignore stay trivial.

## The standard header (REQUIRED, first cell)

Every notebook begins with a markdown cell that gives the title, says
how to open it, and states what the reader will do:

```python
# %% [markdown]
# # <Title, in plain language>
#
# **To open this file as a notebook** in JupyterLab: right-click it
# in the file browser and choose **Open With -> Notebook** (requires
# the `jupytext` extension in the JupyterLab environment). It also
# runs top-to-bottom as a plain Python script.
#
# In this notebook we'll <one or two sentences: what the reader will
# do and on what data/domain>.
```

## Mechanics checklist

- Cells: `# %%` (code) and `# %% [markdown]` (prose; every prose
  line starts `# `).
- Locate the package with the dual-context pattern (script has
  `__file__`; a Jupyter kernel starts in the notebook's directory):

  ```python
  try:
      _here = pl.Path(__file__).parent
  except NameError:
      _here = pl.Path.cwd()
  _pkg = _here.parent
  assert (_pkg / "model.py").exists(), (
      f"expected to run from .../examples, not {_here}"
  )
  sys.path.append(str(_pkg))
  ```

- Imports after that block are fine: `ruff.toml` carries a per-file
  E402 ignore for `examples/`.
- ruff + mypy must pass; line length and naming follow the repo
  conventions (79 cols; doubled loop variables `ii`, `ss`).
- The notebook must run END-TO-END as a script
  (`python examples/<name>.py`) — that run is its test. Guard
  optional data with an existence check and a printed skip message,
  never a crash.
- Keep cells SMALL: one idea per cell, and let the printed/displayed
  output be part of the narrative (print a summary line rather than
  dumping a large object; `display(ds)` when the repr itself teaches).

## Tone: users, not developers

Model the voice on the pywatershed numbered notebooks
(`pywatershed/examples/0*_*.ipynb`). The reader is a hydrologist
trying to get something done, not a maintainer of this code.

Do:

- Write in first-person plural, forward motion: "We'll load the
  control file", "Now we can run the model", "Let's look at the
  output."
- Make every markdown cell set up EXACTLY the next code cell (say
  what we're about to do and why), and interpret notable output in
  the cell after ("We can see the run reached the end time...").
- Anticipate the reader's questions and answer them where they'd
  arise ("That worked! But you may wonder why...").
- Explain terms of art in one clause on first use ("prognostic,
  meaning the next state depends on the current state").
- Invite experimentation ("feel free to increase this to the full
  two years").
- Cross-reference sibling notebooks and point to docstrings/`help()`
  for depth instead of inlining it.
- Close with a short "what just happened / where to go next"
  section; add a References section when citing.

Don't:

- Recount development history, porting decisions, upstream-library
  workarounds, or precision/parity war stories — that material
  belongs in PORTS.md, module docstrings, or code comments; link
  there if a user might genuinely need it.
- Use internal shorthand (arc names, stage numbers, "verbatim",
  "pinned") or defend design choices.
- Front-load a wall of prose: prefer many small markdown cells
  interleaved with code over one long introduction (a compact
  orienting table or list in the header cell is fine).

## Structure template

1. Header cell (above): title + how-to-open + what we'll do.
2. Imports + package location (the dual-context pattern), possibly
   with a short "Prerequisites" note (env, data generation).
3. `##` sections, each a small markdown/code alternation that builds
   the story stepwise; show intermediate results as you go.
4. If the notebook demonstrates a high-level convenience AND its
   details: put the short path FIRST (the one-cell demo a user can
   copy), then an "Under the hood" section that unpacks it.
5. Closing markdown: what just happened, limits, where to go next.
