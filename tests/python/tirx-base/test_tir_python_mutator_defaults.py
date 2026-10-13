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
"""Default Python mutator methods return the native rewritten node."""

import tvm
from tvm import tirx


@tirx.functor.mutator
class Rewrite(tirx.PyStmtExprMutator):
    def visit_int_imm_(self, expr):
        return tirx.IntImm(expr.dtype, int(expr) + 1)

    def visit_add_(self, expr):
        return super().visit_add_(expr)

    def visit_attr_stmt_(self, stmt):
        return super().visit_attr_stmt_(stmt)


def test_default_expression_returns_rewritten_children():
    index = tirx.Var("index", "int32")
    actual = Rewrite().visit_expr(index + 4)
    assert actual is not None
    tvm.ir.assert_structural_equal(actual, index + 5)


def test_default_statement_returns_rewritten_body_and_attribute():
    before = tirx.AttrStmt(0, "preserved", 7, tirx.Evaluate(11))
    actual = Rewrite().visit_stmt(before)
    assert actual is not None
    expected = tirx.AttrStmt(0, "preserved", 8, tirx.Evaluate(12))
    tvm.ir.assert_structural_equal(actual, expected)
