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

import pytest

import tvm
import tvm.ir
import tvm.testing
from tvm import tirx
from tvm.script import tirx as T


def test_simplify_reshape_flattened_index():
    ana = tvm.arith.Analyzer()

    i0 = tirx.Var("i0", "int64")
    i1 = tirx.Var("i1", "int64")
    ana.bind(i0, tvm.ir.Range(0, 8))
    ana.bind(i1, tvm.ir.Range(0, 3))

    i_flattened = i0 * 3 + i1
    tvm.ir.assert_structural_equal(
        ana.simplify((i_flattened) // 12 * 12 + (i_flattened) % 12 // 4 * 4 + (i_flattened) % 4),
        i_flattened,
    )


dtype = tvm.testing.parameter(
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
)


def test_can_prove_self_identity(dtype):
    ana = tvm.arith.Analyzer()

    n = tirx.Var("n", dtype)
    assert ana.can_prove(n == n)


def test_can_prove_self_equal_to_self(dtype):
    ana = tvm.arith.Analyzer()

    n = tirx.Var("n", dtype)
    assert ana.can_prove_equal(n, n)


def test_simplify_symbolic_comparison():
    ana = tvm.arith.Analyzer()

    i0 = tirx.Var("i0", "int64")
    i1 = tirx.Var("i1", "int64")
    n, m = tvm.tirx.SizeVar("n", "int64"), tvm.tirx.SizeVar("m", "int64")
    outer = (n + 31) // 32
    ana.bind(i0, tvm.ir.Range(0, outer))
    ana.bind(i1, tvm.ir.Range(0, 32))
    PS = tvm.arith.ProofStrength

    assert not ana.can_prove(i0 * 32 + i1 < (n + 31) // 32 * 32, PS.DEFAULT)
    assert ana.can_prove(i0 * 32 + i1 < (n + 31) // 32 * 32, PS.SYMBOLIC_BOUND)
    assert ana.can_prove(i0 * 32 + i1 < (n + 31) // 32 * 32 + m, PS.SYMBOLIC_BOUND)
    assert ana.can_prove(i0 * 32 + i1 + 1 <= (n + 31) // 32 * 32, PS.SYMBOLIC_BOUND)
    assert ana.can_prove((n + 31) // 32 * 32 >= i0 * 32 + i1 + 1, PS.SYMBOLIC_BOUND)
    assert ana.can_prove((n + 31) // 32 * 32 >= i0 * 32 + i1, PS.SYMBOLIC_BOUND)


@pytest.mark.parametrize(
    "expression",
    [
        T.vscale() * 32 < T.vscale() * 64,
        T.vscale() * 2 * (T.vscale() * 2) >= T.vscale() * 4,
        (T.vscale() * 4 + 114) // (T.vscale() * 4) * (T.vscale() * 4) >= 115,
        64 % T.vscale() <= T.vscale(),
    ],
)
def test_simplify_vscale_comparison_with_sve_target(expression):
    ana = tvm.arith.Analyzer()

    with tvm.target.Target({"kind": "llvm", "mtriple": "aarch64-linux-gnu", "mattr": ["+sve"]}):
        assert ana.can_prove(expression)


def test_simplify_vscale_comparison_without_sve_target(capfd):
    ana = tvm.arith.Analyzer()
    vs = tvm.tirx.vscale()

    with pytest.raises(AssertionError):
        with tvm.target.Target({"kind": "llvm", "mtriple": "aarch64-linux-gnu"}):
            assert ana.can_prove(vs * 32 < vs * 64)

    warning_prefix = (
        "Warning: The expression contains scalable values. An attempt to prove by substituting "
        "with known values of vscale was not performed. This proof currently only supports "
        "VLA targets, but the target was "
    )
    capture = capfd.readouterr().err
    assert warning_prefix in capture
    assert '"kind":"llvm"' in capture
    assert '"mtriple":"aarch64-linux-gnu"' in capture


def test_regression_simplify_inf_recursion():
    ana = tvm.arith.Analyzer()
    cond = tirx.Var("cond", "int32")

    res = (tvm.tirx.NE(cond, 0).astype("int8") - tvm.tirx.NE(cond, 0).astype("int8")).astype(
        "int32"
    ) == 0
    # regression in a previous case
    # try compare and int set recursive call can cause infinite loop
    ana.rewrite_simplify(res)


def test_bind_allow_override():
    ana = tvm.arith.Analyzer()
    x = tirx.Var("x", "int64")

    ana.bind(x, tvm.ir.Range(0, 10))
    ana.bind(x, tvm.ir.Range(0, 5), allow_override=True)
    assert ana.can_prove(x < 5)

    with pytest.raises(
        tvm.error.TVMError, match="Trying to update var 'x' with a different const bound"
    ):
        ana.bind(x, tvm.ir.Range(0, 3))


def test_simplify_floor_mod_with_linear_offset():
    """
    Test that the floor_mod is simplified correctly when the offset is linear.
    """
    ana = tvm.arith.Analyzer()
    past_decoder_sequence_length = tirx.Var("past_decoder_sequence_length", "int64")
    expr1 = (past_decoder_sequence_length + 1) * 64
    divisor1 = (past_decoder_sequence_length + 1) * 32
    assert ana.can_prove_equal(tvm.tirx.floormod(expr1, divisor1), 0)
    divisor2 = 32 * (past_decoder_sequence_length + 1)
    assert ana.can_prove_equal(tvm.tirx.floormod(expr1, divisor2), 0)


def test_simplify_float_division():
    # Test for the discussion:
    # https://discuss.tvm.apache.org/t/discuss-is-constant-division-to-multiplication-rewrite-in-tvm-necessary/18615
    ana = tvm.arith.Analyzer()
    x = tirx.Var("x", "float32")
    ry = x / 27
    # in old version, the division will be rewritten into x * T.float32(1 / 27)
    sy = ana.rewrite_simplify(ry)
    tvm.ir.assert_structural_equal(ry, sy)


def test_vector_constraint_does_not_crash_z3():
    ana = tvm.arith.Analyzer()
    condition = tirx.Broadcast(tirx.Var("condition", "bool"), 4)
    x = tirx.Var("x", "int32")

    with ana.constraint_scope(condition):
        tvm.ir.assert_structural_equal(ana.simplify(x + 0), x)


def test_z3_context_scope_clone_lifetime():
    x = tirx.Var("x", "int32")

    with tvm.arith.Z3ContextScope():
        analyzer = tvm.arith.Analyzer()
        analyzer.bind(x, tvm.ir.Range.from_min_extent(0, 16))

    # Clone while a different scope is active.  The clone must retain the
    # source Analyzer's context instead of mixing handles from two contexts.
    with tvm.arith.Z3ContextScope():
        cloned = analyzer.clone()

    del analyzer
    assert cloned.can_prove(x >= 0)
    assert cloned.can_prove(x < 16)


@pytest.mark.parametrize("bits, shift", [(8, 1), (32, 12), (64, 63)])
def test_z3_constant_signed_right_shift(bits, shift):
    analyzer = tvm.arith.Analyzer()
    if analyzer.get_smtlib2() == "; Z3 Prover is disabled.":
        pytest.skip("Z3 support is disabled")
    z3 = pytest.importorskip("z3")

    dtype = f"int{bits}"
    x = tirx.Var("x", dtype)
    result = tirx.Var("result", dtype)
    shifted = tirx.shift_right(x + tirx.const(1, dtype), tirx.const(shift, dtype))
    # The dump asserts the negation of its argument, so it defines result == shifted.
    smt = analyzer.get_smtlib2(result != shifted)
    assert "bvashr" not in smt
    solver = z3.Solver()
    solver.from_string(smt)

    highest = (1 << (bits - 1)) - 1
    lowest = -(1 << (bits - 1))
    for value, expected in [(highest, lowest >> shift), (-2, -1), (0, 1 >> shift)]:
        at_value = z3.Int("x") == value
        assert solver.check(at_value, z3.Int("result") == expected) == z3.sat
        assert solver.check(at_value, z3.Int("result") != expected) == z3.unsat


@pytest.mark.parametrize(
    "dtype, mask, values",
    [
        ("int8", 0, [-128, 0, 127]),
        ("int32", 31, [-33, -2, 4, (1 << 31) - 1]),
        ("uint8", 127, [0, 127, 254, 255]),
        ("int64", (1 << 63) - 1, [-(1 << 63), -2, 0, (1 << 63) - 1]),
    ],
)
@pytest.mark.parametrize("mask_first", [False, True])
def test_z3_low_bit_mask(dtype, mask, values, mask_first):
    analyzer = tvm.arith.Analyzer()
    if analyzer.get_smtlib2() == "; Z3 Prover is disabled.":
        pytest.skip("Z3 support is disabled")
    z3 = pytest.importorskip("z3")

    x = tirx.Var("x", dtype)
    result = tirx.Var("result", dtype)
    value = x + tirx.const(1, dtype)
    literal = tirx.const(mask, dtype)
    masked = tirx.bitwise_and(literal, value) if mask_first else tirx.bitwise_and(value, literal)
    smt = analyzer.get_smtlib2(result != masked)
    assert "bvand" not in smt
    solver = z3.Solver()
    solver.from_string(smt)

    for n in values:
        at_value = z3.Int("x") == n
        expected = (n + 1) & mask
        assert solver.check(at_value, z3.Int("result") == expected) == z3.sat
        assert solver.check(at_value, z3.Int("result") != expected) == z3.unsat


@pytest.mark.parametrize("dtype, mask", [("int8", -1), ("uint8", 255), ("int32", 5)])
def test_z3_other_bit_masks_use_bitvectors(dtype, mask):
    analyzer = tvm.arith.Analyzer()
    if analyzer.get_smtlib2() == "; Z3 Prover is disabled.":
        pytest.skip("Z3 support is disabled")
    x = tirx.Var("x", dtype)
    masked = tirx.bitwise_and(x, tirx.const(mask, dtype))
    assert "bvand" in analyzer.get_smtlib2(masked == 0)


def test_z3_shift_does_not_poison_solver():
    analyzer = tvm.arith.Analyzer()
    if analyzer.get_smtlib2() == "; Z3 Prover is disabled.":
        pytest.skip("Z3 support is disabled")

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")

    # Visiting a shift expression must not permanently constrain `b` on the
    # shared solver or contradict an outer scope where `b` is out of range.
    assert not analyzer.can_prove(tirx.shift_left(a, b) >= 0)
    assert not analyzer.can_prove(b >= 0)
    assert not analyzer.can_prove(b < 64)

    with analyzer.constraint_scope(b == -1):
        analyzer.can_prove(tirx.shift_left(a, b) == 0)
        assert not analyzer.can_prove(a == a + 1)


def test_z3_truncdiv_truncmod_semantics():
    analyzer = tvm.arith.Analyzer()
    if analyzer.get_smtlib2() == "; Z3 Prover is disabled.":
        pytest.skip("Z3 support is disabled")
    z3 = pytest.importorskip("z3")

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")

    # TVM Div/Mod truncate toward zero, so negative dividends yield negative remainders.
    assert not analyzer.can_prove(tirx.truncmod(a, 4) >= 0)
    with analyzer.constraint_scope(b != 0):
        assert analyzer.can_prove(tirx.truncdiv(a, b) * b + tirx.truncmod(a, b) == a)

    q = tirx.Var("q", "int32")
    r = tirx.Var("r", "int32")
    smt = analyzer.get_smtlib2(tirx.Or(q != tirx.truncdiv(a, b), r != tirx.truncmod(a, b)))
    solver = z3.Solver()
    solver.from_string(smt)
    for dividend, divisor, exp_q, exp_r in [
        (7, 4, 1, 3),
        (7, -4, -1, 3),
        (-7, 4, -1, -3),
        (-7, -4, 1, -3),
    ]:
        inputs = z3.And(z3.Int("a") == dividend, z3.Int("b") == divisor)
        expected = z3.And(z3.Int("q") == exp_q, z3.Int("r") == exp_r)
        assert solver.check(inputs, expected) == z3.sat
        assert solver.check(inputs, z3.Not(expected)) == z3.unsat


if __name__ == "__main__":
    tvm.testing.main()
