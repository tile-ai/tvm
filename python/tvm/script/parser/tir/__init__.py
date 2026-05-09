"""Backward-compatibility shim: tvm.script.parser.tir -> tvm.tirx.script.parser"""
from tvm.tirx.script.parser import *  # noqa: F401,F403
from tvm.tirx.script.parser.entry import *  # noqa: F401,F403
