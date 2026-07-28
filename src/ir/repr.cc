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

/*!
 * \file ir/repr.cc
 * \brief Implements Dump helpers and FFI registration for ffi-repr-based printing.
 *
 * The legacy ReprPrinter has been replaced by ffi::ReprPrint.  This file:
 *  - Implements the Dump() debug helpers (they call ffi::ReprPrint).
 *  - Registers node.AsRepr (for backward Python compatibility) via ffi::ReprPrint.
 *
 * Note: __ffi_repr__ hooks for ffi::reflection::AccessPath and AccessStep are
 * registered by tvm-ffi itself.  Registering them again here causes a
 * double-registration abort when loading against tvm-ffi 0.1.12 or newer.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/repr.h>
#include <tvm/runtime/device_api.h>

namespace tvm {

void Dump(const ffi::ObjectRef& n) { std::cerr << ffi::ReprPrint(ffi::Any(n)) << "\n"; }

void Dump(const ffi::Object* n) { Dump(ffi::GetRef<ffi::ObjectRef>(n)); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // node.AsRepr: backward-compatible Python entry point.
  // Python's tvm.runtime._ffi_node_api sets __object_repr__ = AsRepr via init_ffi_api.
  refl::GlobalDef().def("node.AsRepr",
                        [](ffi::Any obj) -> ffi::String { return ffi::ReprPrint(obj); });
}
}  // namespace tvm
