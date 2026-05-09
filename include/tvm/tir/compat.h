#ifndef TVM_TIR_COMPAT_H_
#define TVM_TIR_COMPAT_H_

#include <tvm/ffi/error.h>

#ifndef ICHECK
#define ICHECK TVM_FFI_ICHECK
#define ICHECK_EQ TVM_FFI_ICHECK_EQ
#define ICHECK_NE TVM_FFI_ICHECK_NE
#define ICHECK_LT TVM_FFI_ICHECK_LT
#define ICHECK_GT TVM_FFI_ICHECK_GT
#define ICHECK_LE TVM_FFI_ICHECK_LE
#define ICHECK_GE TVM_FFI_ICHECK_GE
#endif

namespace tvm {
namespace tir = tirx;
}

#endif  // TVM_TIR_COMPAT_H_
