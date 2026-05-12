# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Generic opertors in TVM.
We follow the numpy naming convention for this interface
(e.g., tvm.tir.generic.multitply ~ numpy.multiply).
The default implementation is used by tvm.ExprOp.
"""
# pylint: disable=unused-argument
from . import _ffi_api

# Operator precedence used when overloading.
__op_priority__ = 0


def add(lhs, rhs, span=None):
    """Generic add operator.

    Parameters
    ----------
    lhs : object
        The left operand.
    rhs : object
        The right operand.
    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    op : tvm.Expr
        The result Expr of add operaton.
    """
    return _ffi_api._OpAdd(lhs, rhs, span)  # type: ignore


def subtract(lhs, rhs, span=None):
    """Generic subtract operator.

    Parameters
    ----------
    lhs : object
        The left operand.
    rhs : object
        The right operand.
    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    op : tvm.Expr
        The result Expr of subtract operaton.
    """
    return _ffi_api._OpSub(lhs, rhs, span)  # type: ignore


def multiply(lhs, rhs, span=None):
    """Generic multiply operator.

    Parameters
    ----------
    lhs : object
        The left operand.
    rhs : object
        The right operand.
    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    op : tvm.Expr
        The result Expr of multiply operaton.
    """
    return _ffi_api._OpMul(lhs, rhs, span)  # type: ignore


def divide(lhs, rhs, span=None):
    """Generic divide operator.

    Parameters
    ----------
    lhs : object
        The left operand.
    rhs : object
        The right operand.
    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    op : tvm.Expr
        The result Expr of divide operaton.
    """
    return _ffi_api._OpDiv(lhs, rhs, span)  # type: ignore


def floordiv(lhs, rhs, span=None):
    """Generic floordiv operator.

    Parameters
    ----------
    lhs : object
        The left operand.
    rhs : object
        The right operand.
    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    op : tvm.Expr
        The result Expr of floordiv operaton.
    """
    return _ffi_api._OpFloorDiv(lhs, rhs, span)  # type: ignore


_VALID_CAST_ROUNDING_MODES = {"", "rn", "rz", "rp", "rm", "rs"}


def cast(src, dtype, round="", sat=True, rbits=None, span=None):
    """Generic cast operator.

    Parameters
    ----------
    src : object
        The source operand.
    dtype : str
        The target data type.
    round : str, optional
        Rounding mode (e.g. "rn", "rz", "rp", "rm", "rs").
        Empty string means use backend default.
    sat : bool, optional
        Saturate to finite (True = PTX .satfinite, default).
    rbits : PrimExpr, optional
        Random bits operand for stochastic rounding (round="rs").
    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    op : tvm.Expr
        The result Expr of cast operaton.

    Notes
    -----
    Internally, ``round``/``sat``/``rbits`` are stored as ``"tl.round"``,
    ``"tl.sat"``, and ``"tl.rbits"`` keys in the CastNode's ``annotations``
    map (mirroring the annotations pattern on Call/For/Block/Allocate).
    """
    if round not in _VALID_CAST_ROUNDING_MODES:
        raise ValueError(
            f"Invalid round '{round}'. "
            f"Must be one of: {sorted(_VALID_CAST_ROUNDING_MODES)}"
        )
    if not isinstance(sat, bool):
        raise ValueError(
            f"Invalid sat '{sat}'. Must be a bool (True for satfinite, False for no saturation)"
        )
    if round == "rs" and rbits is None:
        raise ValueError("rbits is required when round='rs' (stochastic rounding)")
    if round != "rs" and rbits is not None:
        raise ValueError("rbits is only valid with round='rs' (stochastic rounding)")
    # Local import to avoid a circular dependency with .expr at module load time.
    from .expr import IntImm, StringImm
    annotations = {}
    if round:
        annotations["tl.round"] = StringImm(round)
    if not sat:
        annotations["tl.sat"] = IntImm("bool", 0)
    if rbits is not None:
        annotations["tl.rbits"] = rbits
    return _ffi_api._cast(dtype, src, annotations or None, span)  # type: ignore
