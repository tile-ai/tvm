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
#include <tvm/tirx/op.h>

#include <limits>

namespace tvm::arith {
namespace {

using namespace tirx;

class Z3ProverTest : public ::testing::Test {
 protected:
  void SetUp() override {
    Analyzer probe;
    if (!probe.z3_prover.CanProve(Bool(true))) {
      GTEST_SKIP() << "Z3 support is disabled";
    }
  }

  void Bind(const Var& var, int64_t min, int64_t extent) {
    analyzer_.Bind(
        var, Range::FromMinExtent(make_const(var.dtype(), min), make_const(var.dtype(), extent)));
  }

  Analyzer analyzer_;
};

TEST_F(Z3ProverTest, DenseCountWithinBudget) {
  Var tx("tx", DataType::Int(32));
  Bind(tx, 0, 1024);
  analyzer_.z3_prover.SetRLimit(1000);
  EXPECT_EQ(analyzer_.z3_prover.CountSatisfyingValues(tx, 1024), 1024);
  EXPECT_EQ(analyzer_.z3_prover.CountSatisfyingValues(tx, 1024), 1024);
}

TEST_F(Z3ProverTest, CountWithPossiblyEmptyLoop) {
  Var bx("bx", DataType::Int(32)), start("start", DataType::Int(32));
  Var token("token", DataType::Int(32));
  Bind(bx, 0, 64);
  Bind(start, 0, 256);
  // The loop is empty for start >= 1, which leaves token unconstrained.
  analyzer_.Bind(token, Range::FromMinExtent(start, 1 - start));
  analyzer_.z3_prover.SetRLimit(10000);
  EXPECT_EQ(analyzer_.z3_prover.CountSatisfyingValues(start, 256), 256);
}

TEST_F(Z3ProverTest, ProjectedValuesAndConsecutiveGroups) {
  Var tx("tx", DataType::Int(32)), y("y", DataType::Int(32));
  Bind(tx, 0, 256);
  Bind(y, 0, 2);
  {
    With<ConstraintContext> scope(&analyzer_, floormod(tx, 4) == y);
    EXPECT_EQ(analyzer_.z3_prover.CountSatisfyingValues(tx, 256, 2), 128);
    EXPECT_EQ(analyzer_.z3_prover.CountSatisfyingValues(tx, 256, 3), -2);
  }
  EXPECT_EQ(analyzer_.z3_prover.CountSatisfyingValues(tx, 256, 32), 256);
}

TEST_F(Z3ProverTest, SparseWarpGroups) {
  Var tx("tx", DataType::Int(32));
  Bind(tx, 0, 256);
  {
    With<ConstraintContext> scope(&analyzer_, floormod(floordiv(tx, 32), 2) == 0);
    EXPECT_EQ(analyzer_.z3_prover.CountSatisfyingValues(tx, 256, 32), 128);
  }
  {
    With<ConstraintContext> scope(&analyzer_, floormod(tx, 4) == 0);
    EXPECT_EQ(analyzer_.z3_prover.CountSatisfyingValues(tx, 256), 64);
    EXPECT_EQ(analyzer_.z3_prover.CountSatisfyingValues(tx, 256, 32), -2);
  }
}

TEST_F(Z3ProverTest, OffsetAndWideSparseDomains) {
  Var value("value", DataType::Int(64));
  for (int64_t min : {int64_t{-64}, int64_t{1} << 40, std::numeric_limits<int64_t>::min() + 2,
                      std::numeric_limits<int64_t>::max() - 256}) {
    Analyzer analyzer;
    analyzer.Bind(value, Range::FromMinExtent(make_const(value.dtype(), min),
                                              make_const(value.dtype(), 256)));
    With<ConstraintContext> values(&analyzer, floormod(value, 4) == 1);
    EXPECT_EQ(analyzer.z3_prover.CountSatisfyingValues(value, 256), 64);
  }
  Bind(value, 0, int64_t{1} << 20);
  With<ConstraintContext> scope(&analyzer_, floormod(value, 65536) == 0);
  EXPECT_EQ(analyzer_.z3_prover.CountSatisfyingValues(value, 32), 16);
}

TEST_F(Z3ProverTest, SmallSolutionSetAndEmptyDomain) {
  Var value("value", DataType::Int(32));
  Bind(value, 0, 256);
  {
    With<ConstraintContext> scope(&analyzer_, value * value < 40);
    EXPECT_EQ(analyzer_.z3_prover.CountSatisfyingValues(value, 256), 7);
  }
  {
    With<ConstraintContext> scope(&analyzer_, value < 0);
    EXPECT_EQ(analyzer_.z3_prover.CountSatisfyingValues(value, 256), 0);
  }
  EXPECT_EQ(analyzer_.z3_prover.CountSatisfyingValues(value, 0), 0);
  EXPECT_EQ(analyzer_.z3_prover.CountSatisfyingValues(value, 32), 32);
}

TEST_F(Z3ProverTest, CappedWideAndUnboundedDomains) {
  Var value("value", DataType::Int(64));
  EXPECT_EQ(analyzer_.z3_prover.CountSatisfyingValues(value, 17), 17);
  auto lower = make_const(value.dtype(), std::numeric_limits<int64_t>::min() + 2);
  auto upper = make_const(value.dtype(), std::numeric_limits<int64_t>::max() - 1);
  With<ConstraintContext> scope(&analyzer_, value >= lower && value <= upper);
  EXPECT_EQ(analyzer_.z3_prover.CountSatisfyingValues(value, 17), 17);
}

TEST_F(Z3ProverTest, UnknownIsNotAnExactCount) {
  Var tx("tx", DataType::Int(32));
  Bind(tx, 0, 256);
  analyzer_.z3_prover.SetRLimit(1);
  EXPECT_EQ(analyzer_.z3_prover.CountSatisfyingValues(tx, 256), -1);
  analyzer_.z3_prover.SetRLimit(10000);
  EXPECT_EQ(analyzer_.z3_prover.CountSatisfyingValues(tx, 256), 256);
  EXPECT_FALSE(analyzer_.z3_prover.CanProve(tx != 0));
}

}  // namespace
}  // namespace tvm::arith
