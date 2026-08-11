"""prms_translate: legacy PRMS files -> pws_phoenix model inputs.

ONE-WAY street (PRMS native -> pws_phoenix; never back) and modularly
DISTINCT from the core: the core framework/process code never imports
this package (the dependency is strictly prms_translate -> core).

DEPENDENCY RULE: file-format decoding leans on **pyPRMS** exclusively
(plus numpy/xarray) -- NEVER pywatershed. The intent (JLM, Aug 2026)
is that the generic PRMS-file capabilities here migrate INTO pyPRMS
over time, so they must not drag pywatershed along:

- migratable-to-pyPRMS half: the float64-parse shim + monthly-dim
  convention (readers.py), the dynamic-parameter reader
  (dyn_param.py), CBH windowing/widening (cbh.py).
- pws_phoenix-specific half (stays here): control-driven CLASS
  resolution (control.py) and contract-driven parameter packaging.

Upstream asks for pyPRMS (evaluated Aug 2026 on 0.9.10, the latest
release; see readers.py/dyn_param.py for the details):

1. an option to parse PRMS type-F parameters as float64
   (ParameterFile parses them float32 at the text line -- a ~1e-9
   truncation that breaks bitwise parity with a float64 parse);
2. a dynamic-parameter (.param/.dyn) file reader.
"""

from prms_translate.assemble import (
    ModelKit,
    assemble_from_control,
    model_from_control,
)
from prms_translate.assemble_mpi import (
    mpi_model_from_control,
    write_mpi_input_file,
)
from prms_translate.cbh import load_cbh
from prms_translate.control import (
    PrmsRunConfig,
    from_control,
    resolve_classes,
)
from prms_translate.dyn_param import load_dynamic_parameter
from prms_translate.preprocess import (
    digest_array,
    verify_preprocessed,
    write_preprocessed,
)
from prms_translate.parameters import (
    package_parameters,
    volume_map_weights,
)
from prms_translate.readers import (
    load_control,
    load_parameters,
    prms_metadata,
)

__all__ = [
    "ModelKit",
    "PrmsRunConfig",
    "assemble_from_control",
    "from_control",
    "model_from_control",
    "load_cbh",
    "load_control",
    "load_dynamic_parameter",
    "load_parameters",
    "mpi_model_from_control",
    "digest_array",
    "package_parameters",
    "prms_metadata",
    "resolve_classes",
    "verify_preprocessed",
    "volume_map_weights",
    "write_mpi_input_file",
    "write_preprocessed",
]
