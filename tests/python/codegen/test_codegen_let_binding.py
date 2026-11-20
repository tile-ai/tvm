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

"""Regression tests for per-function ``CodeGenC`` state."""

import tvm
import tvm.testing
from tvm import tirx


def test_let_binding_state_is_function_scoped():
    """A Let Var may be reused with different values across functions.

    ``CodeGenC::let_binding_`` rejects the same Var being bound to two
    different values inside one function. The map must not leak across
    functions: clients such as the Ascend SuperKernel emitter re-emit a
    cloned body and rebind the original Var objects with substituted values.
    """
    v = tirx.Var("v", "int32")

    def make_func(val):
        body = tirx.Let(v, tirx.IntImm("int32", val), v + tirx.IntImm("int32", 1))
        return tirx.PrimFunc([], tirx.Evaluate(body))

    mod = tvm.IRModule({"f1": make_func(1), "f2": make_func(2)})
    build_c = tvm.ffi.get_global_func("target.build.c")
    # Before the fix this raised "Let cannot bind the same var to two
    # different values" while emitting the second function.
    build_c(mod, tvm.target.Target("c"))


if __name__ == "__main__":
    tvm.testing.main()
