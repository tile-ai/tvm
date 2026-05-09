#ifndef TVM_TARGET_TARGET_INFO_H_
#define TVM_TARGET_TARGET_INFO_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>

#include <string>

namespace tvm {

class MemoryInfoNode : public Object {
 public:
  int64_t unit_bits;
  int64_t max_num_bits;
  int64_t max_simd_bits;
  PrimExpr head_address;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MemoryInfoNode>()
        .def_ro("unit_bits", &MemoryInfoNode::unit_bits)
        .def_ro("max_num_bits", &MemoryInfoNode::max_num_bits)
        .def_ro("max_simd_bits", &MemoryInfoNode::max_simd_bits)
        .def_ro("head_address", &MemoryInfoNode::head_address);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("target.MemoryInfo", MemoryInfoNode, Object);
};

class MemoryInfo : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(MemoryInfo, ObjectRef, MemoryInfoNode);
};

TVM_DLL MemoryInfo GetMemoryInfo(const std::string& scope);

}  // namespace tvm
#endif  // TVM_TARGET_TARGET_INFO_H_
