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

#include "runtime/pack_args.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <utility>

namespace tvm {
namespace runtime {

TEST(PackArgs, BFloat16VoidAddress) {
  uint16_t result = 0;
  auto packed =
      PackFuncVoidAddr([&](ffi::PackedArgs, ffi::Any*,
                           void** args) { std::memcpy(&result, args[0], sizeof(result)); },
                       {DLDataType{kDLBfloat, 16, 1}});
  // Non-exact values, both tie directions, signed zero, and non-finite values.
  for (auto test : {std::pair<double, uint16_t>{1.5, 0x3fc0},
                    {-1.1, 0xbf8d},
                    {1.00390625, 0x3f80},
                    {1.01171875, 0x3f82},
                    {-0.0, 0x8000},
                    {std::numeric_limits<double>::infinity(), 0x7f80},
                    {std::numeric_limits<double>::quiet_NaN(), 0x7fc0}}) {
    packed(test.first);
    EXPECT_EQ(result, test.second);
  }
}

TEST(PackArgs, BFloat16NonBufferArgument) {
  auto packed = PackFuncNonBufferArg(
      [](ffi::PackedArgs, ffi::Any*, ArgUnion64* args) {
        EXPECT_EQ(args[0].v_uint16[0], 0x3fc0);
        EXPECT_FLOAT_EQ(args[1].v_float32[0], 2.5f);
      },
      {DLDataType{kDLOpaqueHandle, 64, 1}, DLDataType{kDLBfloat, 16, 1},
       DLDataType{kDLFloat, 32, 1}});
  packed(static_cast<void*>(nullptr), 1.5, 2.5);
}

TEST(PackArgs, BFloat16AlignedArguments) {
  struct Arguments {
    uint16_t first;
    uint16_t second;
    int32_t integer;
    uint16_t third;
    double real;
    void* pointer;
  };
  int value = 0;
  auto packed = PackFuncPackedArgAligned(
      [&](ffi::PackedArgs, ffi::Any*, void* data, size_t nbytes) {
        ASSERT_EQ(nbytes, sizeof(Arguments));
        Arguments args;
        std::memcpy(&args, data, sizeof(args));
        EXPECT_EQ(args.first, 0x3fc0);
        EXPECT_EQ(args.second, 0xbf8d);
        EXPECT_EQ(args.integer, 42);
        EXPECT_EQ(args.third, 0x4000);
        EXPECT_DOUBLE_EQ(args.real, 3.5);
        EXPECT_EQ(args.pointer, &value);
      },
      {DLDataType{kDLBfloat, 16, 1}, DLDataType{kDLBfloat, 16, 1}, DLDataType{kDLInt, 32, 1},
       DLDataType{kDLBfloat, 16, 1}, DLDataType{kDLFloat, 64, 1},
       DLDataType{kDLOpaqueHandle, 64, 1}});
  packed(1.5, -1.1, 42, 2.0, 3.5, static_cast<void*>(&value));
}

}  // namespace runtime
}  // namespace tvm
