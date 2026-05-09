# isort: skip_file
"""Backward-compatibility shim: tvm.tir re-exports from tvm.tirx."""
from tvm.tirx import *  # noqa: F401,F403
from tvm.tirx import _ffi_api  # noqa: F401

from tvm.tirx import transform  # noqa: F401
from tvm.tirx import analysis  # noqa: F401
from tvm.tirx import backend  # noqa: F401
from tvm.tirx import stmt_functor  # noqa: F401

import tvm.script

tvm.script.register_dialect("tir", "tvm.tirx.script")
