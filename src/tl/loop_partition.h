/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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

 /*!
  * \file loop_partition.h
  * \brief Partition parallel loops onto threads
  */

#ifndef TVM_TL_LOOP_PARTITION_H_
#define TVM_TL_LOOP_PARTITION_H_

#include <tvm/tir/op.h>

#include "layout.h"

namespace tvm {
namespace tl {

using namespace tir;

For PartitionLoop(const ForNode* op, const Var& thread, arith::Analyzer *analyzer, const Fragment& loop_layout);

For PartitionLoop(const ForNode* op, const Var& thread, arith::Analyzer *analyzer, size_t num_thread);

Stmt LoopPragmaUnroll(Stmt stmt);

} // namespace tl
} // namespace tvm


#endif // TVM_TL_LOOP_PARTITION_H_
