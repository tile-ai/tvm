/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <gtest/gtest.h>
#include <tvm/arith/analyzer.h>
#include <tvm/ir/with_context.h>
#include <tvm/tirx/op.h>

namespace tvm::arith {

TEST(Z3Prover, DistinguishesUnsatFromUnknown) {
  Analyzer availability_probe;
  if (availability_probe.z3_prover.GetStats() == "; Z3 Prover is disabled.") {
    GTEST_SKIP() << "Z3 support is disabled";
  }

  tirx::Var var("value", DataType::Int(32));
  Range domain =
      Range::FromMinExtent(tirx::make_const(var.dtype(), 0), tirx::make_const(var.dtype(), 4));

  Analyzer unsat_analyzer;
  unsat_analyzer.Bind(var, domain);
  {
    With<ConstraintContext> constraint(&unsat_analyzer, var < 0);
    EXPECT_EQ(unsat_analyzer.z3_prover.CountSatisfyingValues(var, 4), 0);
  }

  Analyzer unknown_analyzer;
  unknown_analyzer.Bind(var, domain);
  unknown_analyzer.z3_prover.SetRLimit(1);
  EXPECT_EQ(unknown_analyzer.z3_prover.CountSatisfyingValues(var, 4), -1);
}

}  // namespace tvm::arith
