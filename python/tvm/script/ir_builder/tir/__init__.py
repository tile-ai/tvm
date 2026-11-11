"""Backward-compatibility shim: tvm.script.ir_builder.tir -> tvm.tirx.script.builder"""
from tvm.tirx.script.builder import *  # noqa: F401,F403
from tvm.tirx.script.builder import ir  # noqa: F401
from tvm.tirx.script.builder import _ffi_api  # noqa: F401
from tvm.tirx.script.builder.frame import *  # noqa: F401,F403
from tvm.tirx.script.builder.utils import buffer_proxy, frame_scope, seq_scope  # noqa: F401
