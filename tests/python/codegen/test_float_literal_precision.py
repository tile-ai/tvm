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
import re

import numpy as np
import pytest

import tvm
from tvm import tirx


@pytest.mark.parametrize(
    "target,dtype", [("c", "float32"), ("c", "float64"), ("metal", "float32"), ("metal", "float16")]
)
@pytest.mark.parametrize("value", [1.2345678901234567, -1.2345678901234567, 0.00012345678901234567])
def test_float_literal_round_trip(target, dtype, value):
    """Source literals preserve the value held by FloatImm, without a device."""
    output = tirx.decl_buffer((1,), dtype, name="output")
    constant = tirx.FloatImm(dtype, value)
    body = tirx.BufferStore(output, constant, [0])
    function = tirx.PrimFunc([output.data], body).with_attr("global_symbol", "main")
    function = function.with_attr("tirx.noalias", True)
    if target == "metal":
        function = function.with_attr("calling_conv", 2)
        function = function.with_attr("tirx.kernel_launch_params", [])
    module = tvm.get_global_func("target.build." + target)(
        tvm.IRModule({"main": function}), tvm.target.Target(target)
    )
    source = module.inspect_source()
    literals = re.findall(r"-?\d+\.\d+e[+-]\d+", source)
    assert literals, source
    parsed = np.dtype(dtype).type(float(literals[-1]))
    expected = np.dtype(dtype).type(constant.value)
    assert parsed.tobytes() == expected.tobytes(), source
