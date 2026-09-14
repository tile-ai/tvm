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
"""Tests for Analyzer.bind_lazy_bounds: definition-backed bounds that are
evaluated under the constraints active at query time, instead of being
snapshotted at bind time (motivated by tile-ai/tilelang#3220)."""
import pytest

import tvm
import tvm.testing
from tvm import tirx
from tvm.arith import Analyzer
from tvm.ir import Range


def _range(lo, hi):
    return Range.from_min_extent(tirx.IntImm("int32", lo), tirx.IntImm("int32", hi - lo))


def test_eager_bind_snapshots_stale_bound():
    # Documents the premature-evaluation behavior lazy binds exist to avoid:
    # v = tx bound before the constraint, snapshot never tightens.
    analyzer = Analyzer()
    tx = tirx.Var("tx", "int32")
    v = tirx.Var("v", "int32")
    analyzer.bind(tx, _range(0, 256))
    analyzer.bind(v, tx)
    with analyzer.constraint_scope(tx < 64):
        assert analyzer.const_int_bound(tx).max_value == 63
        assert analyzer.const_int_bound(v).max_value == 255  # stale snapshot


def test_lazy_bind_sees_later_constraints():
    analyzer = Analyzer()
    tx = tirx.Var("tx", "int32")
    v = tirx.Var("v", "int32")
    analyzer.bind(tx, _range(0, 256))
    analyzer.bind_lazy_bounds(v, tx)
    assert analyzer.const_int_bound(v).max_value == 255
    with analyzer.constraint_scope(tx < 64):
        assert analyzer.const_int_bound(v).max_value == 63
    # scope exited: the memoized tightened bound must not leak
    assert analyzer.const_int_bound(v).max_value == 255


def test_lazy_bind_is_order_insensitive():
    # Bind after the constraint scope opened vs before: same answer.
    analyzer = Analyzer()
    tx = tirx.Var("tx", "int32")
    v = tirx.Var("v", "int32")
    analyzer.bind(tx, _range(0, 256))
    with analyzer.constraint_scope(tx < 64):
        analyzer.bind_lazy_bounds(v, tx)
        assert analyzer.const_int_bound(v).max_value == 63


def test_lazy_definition_chain():
    analyzer = Analyzer()
    tx = tirx.Var("tx", "int32")
    v = tirx.Var("v", "int32")
    w = tirx.Var("w", "int32")
    analyzer.bind(tx, _range(0, 256))
    analyzer.bind_lazy_bounds(v, tx)
    analyzer.bind_lazy_bounds(w, v * 2)
    with analyzer.constraint_scope(tx < 64):
        assert analyzer.const_int_bound(w).max_value == 126
    assert analyzer.const_int_bound(w).max_value == 510


def test_lazy_bind_can_prove_through_simplify():
    # CanProve routes through simplification; the rewrite rule plus the
    # lazy bound must agree.
    analyzer = Analyzer()
    tx = tirx.Var("tx", "int32")
    v = tirx.Var("v", "int32")
    analyzer.bind(tx, _range(0, 256))
    analyzer.bind_lazy_bounds(v, tx)
    with analyzer.constraint_scope(tx < 64):
        assert analyzer.can_prove(v < 64)


def test_lazy_bind_rejects_silent_override():
    analyzer = Analyzer()
    tx = tirx.Var("tx", "int32")
    v = tirx.Var("v", "int32")
    analyzer.bind(tx, _range(0, 256))
    analyzer.bind_lazy_bounds(v, tx)
    with pytest.raises(tvm.error.TVMError):
        analyzer.bind_lazy_bounds(v, tx + 1)
    analyzer.bind_lazy_bounds(v, tx + 1, allow_override=True)
    assert analyzer.const_int_bound(v).max_value == 256


def test_nested_constraint_scopes_invalidate_memo():
    analyzer = Analyzer()
    tx = tirx.Var("tx", "int32")
    v = tirx.Var("v", "int32")
    analyzer.bind(tx, _range(0, 256))
    analyzer.bind_lazy_bounds(v, tx)
    with analyzer.constraint_scope(tx < 128):
        assert analyzer.const_int_bound(v).max_value == 127
        with analyzer.constraint_scope(tx < 32):
            assert analyzer.const_int_bound(v).max_value == 31
        assert analyzer.const_int_bound(v).max_value == 127


if __name__ == "__main__":
    tvm.testing.main()
