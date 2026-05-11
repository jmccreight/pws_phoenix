<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [MRE: `PWSAccessor` and `process_factory` Design Discussion](#mre-pwsaccessor-and-process_factory-design-discussion)
  - [What the MRE demonstrates](#what-the-mre-demonstrates)
    - [Section 1 — `PWSAccessor` (registered once, globally)](#section-1--pwsaccessor-registered-once-globally)
    - [Section 2 — `process_factory`](#section-2--process_factory)
    - [Section 3 — Two spec classes with identical structure, different behaviour](#section-3--two-spec-classes-with-identical-structure-different-behaviour)
    - [Section 4 — Demo output](#section-4--demo-output)
  - [The design trade-off worth discussing](#the-design-trade-off-worth-discussing)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# MRE: `PWSAccessor` and `process_factory` Design Discussion

The file `mre_accessor_design.py` is a minimal, self-contained example
illustrating the two key design ideas in `base_attrs.py`. No NumPy arrays, no
file I/O, no real hydrology — just plain Python/xarray.

---

## What the MRE demonstrates

The file has four clearly separated sections.

### Section 1 — `PWSAccessor` (registered once, globally)

```py
@xr.register_dataset_accessor("pws")
class PWSAccessor:
    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds

    def advance(self) -> None:
        self._ds.attrs["advance"](self._ds)

    def calculate(self, dt: float) -> None:
        self._ds.attrs["calculate"](self._ds, dt)
```

The accessor class is a singleton registered with xarray. It has **no idea
what process it's attached to** — it just reaches into `ds.attrs` and calls
whatever callable was stashed there. This is the "curious" bit: `ds.attrs` is
normally for scalar metadata (units, source, etc.), but here it stores live
Python callables. xarray doesn't restrict the value type, so this works, even
though it's unconventional.

### Section 2 — `process_factory`

```py
def process_factory(spec_cls, **init_values) -> xr.Dataset:
    ...
    ds.attrs["advance"]    = spec_cls.advance     # <-- behaviour varies here
    ds.attrs["calculate"]  = spec_cls.calculate   # <-- and here
    return ds
```

The factory stamps the *spec class's* static methods into `ds.attrs` at
construction time. From that point on, `ds.pws.advance()` calls `Decay.advance`
or `Growth.advance` depending on which spec was used — not via inheritance or
`isinstance` checks, but via a plain dictionary lookup.

### Section 3 — Two spec classes with identical structure, different behaviour

`Decay` and `Growth` share the same field names (`rate`, `value`, `value_prev`)
but implement `advance`/`calculate` differently. Neither inherits from anything,
and neither knows about the accessor. The decoupling is total.

### Section 4 — Demo output

```
step   decay.value  growth.value
----------------------------------
   1       60.6531        3.0000
   2       36.7879        6.0000
   3       22.3130        9.0000
   4       13.5335       12.0000
```

---

## The design trade-off worth discussing

The core tension is that **`ds.attrs` is being used as a dispatch table**, not
as metadata. Some implications:

| Concern | Notes |
|---|---|
| **Serialization** | `xr.Dataset.to_netcdf()` will silently drop the callable attrs (NetCDF can't store functions). The dataset cannot round-trip through disk in its "live" form. |
| **Type safety** | Nothing prevents `ds.attrs["advance"]` from being `None` or missing — the `KeyError` only surfaces at call time. |
| **Introspection** | `ds` printed in a notebook looks like a normal dataset; the hidden callables are invisible unless you know to look in `attrs`. |
| **Flexibility gained** | Any two datasets with the same variable names but different spec classes behave differently through the *identical* accessor interface, with zero subclassing. This is duck-typing at the dataset level. |
