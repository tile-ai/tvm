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
 * \file cuda_module.cc
 * \brief CUDAModuleNode — runtime-side, plugin-only.  Reachable from C++ only
 *        through the FFI registry keys "ffi.Module.create.cuda" and
 *        "ffi.Module.load_from_bytes.cuda".  No exported header — codegen-side
 *        construction goes through src/target/cuda/cuda_fallback_module.h.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <array>
#include <mutex>
#include <string>
#include <vector>

#include "../../support/bytes_io.h"
#include "../metadata.h"
#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "cuda_common.h"

namespace tvm {
namespace runtime {
namespace {

inline void EnsureCurrentDeviceContext(int device_id) {
  // Driver API entry points require a current context on this thread. `cudaGetDevice`
  // reports the logical device, but it does not guarantee the primary context is bound.
  CUDA_CALL(cudaSetDevice(device_id));
}

}  // namespace

// Maximum number of GPU supported in CUDAModule (file-local).
static constexpr const int kMaxNumGPUs = 32;

// Module to support thread-safe multi-GPU execution.
// cuModule is a per-GPU module
// The runtime will contain a per-device module table
// The modules will be lazily loaded
class CUDAModuleNode : public ffi::ModuleObj {
 public:
  CUDAModuleNode(ffi::Bytes code, ffi::String fmt, ffi::Map<ffi::String, FunctionInfo> fmap,
                 ffi::Map<ffi::String, ffi::String> source)
      : code_(code), fmt_(fmt), fmap_(fmap), source_(source) {
    std::fill(module_.begin(), module_.end(), nullptr);
  }
  // destructor
  ~CUDAModuleNode() {
    for (size_t i = 0; i < module_.size(); ++i) {
      if (module_[i] != nullptr) {
        cudaError_t set_err = cudaSetDevice(static_cast<int>(i));
        if (set_err != cudaSuccess && set_err != cudaErrorCudartUnloading) {
          continue;
        }
        CUresult result = cuModuleUnload(module_[i]);
        // Ignore errors during cleanup - context may be shutting down
        (void)result;
      }
    }
  }

  const char* kind() const final { return "cuda"; }

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final {
    return ffi::Module::kBinarySerializable | ffi::Module::kRunnable;
  }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final;

  ffi::Bytes SaveToBytes() const final {
    // Format: [fmt][fmap][code].  Source map is in-memory inspection only and
    // is NEVER serialized — it is lost on save/load round-trip (matches
    // upstream behavior; the receiver rebuilds source from code bytes if
    // possible).  CUDAFallbackModuleNode::SaveToBytes (in
    // src/target/cuda/cuda_fallback_module.cc) MUST mirror this format
    // byte-for-byte; see one-way comment there.
    std::string buffer;
    support::BytesOutStream stream(&buffer);
    stream.Write(fmt_);
    stream.Write(fmap_);
    stream.Write(code_);
    return ffi::Bytes(std::move(buffer));
  }

  ffi::String InspectSource(const ffi::String& format) const final {
    // For known compiled formats, return code as string when format matches.
    if (format == fmt_) {
      return ffi::String(code_.data(), code_.size());
    }
    // Look up the source map for an exact match (e.g. "cuda" returns the
    // original C++ source, populated by codegen at construction time).
    if (auto it = source_.find(format); it != source_.end()) {
      return (*it).second;
    }
    // Empty-format (`mod.get_source()`) — prefer the `cuda` source if present,
    // else fall back to code-as-string when fmt_ is textual.
    if (format.empty()) {
      if (auto it = source_.find("cuda"); it != source_.end()) {
        return (*it).second;
      }
      if (fmt_ == "ptx" || fmt_ == "cuda") {
        return ffi::String(code_.data(), code_.size());
      }
    }
    return ffi::String();
  }

  // get a CUfunction from primary context in device_id
  CUfunction GetFunc(int device_id, const std::string& func_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    EnsureCurrentDeviceContext(device_id);
    // must recheck under the lock scope
    if (module_[device_id] == nullptr) {
      CUDA_DRIVER_CALL(cuModuleLoadData(&(module_[device_id]), code_.data()));
      static auto nvshmem_init_hook = ffi::Function::GetGlobal("runtime.nvshmem.cumodule_init");
      if (nvshmem_init_hook.has_value()) {
        (*nvshmem_init_hook)(static_cast<void*>(module_[device_id]));
      }
    }
    CUfunction func;
    CUresult result = cuModuleGetFunction(&func, module_[device_id], func_name.c_str());
    if (result != CUDA_SUCCESS) {
      const char* msg;
      cuGetErrorName(result, &msg);
      TVM_FFI_THROW(CUDAError) << "cuModuleGetFunction " << func_name
                               << " failed with error: " << msg;
    }
    return func;
  }
  // get a global var from primary context in device_id
  CUdeviceptr GetGlobal(int device_id, const std::string& global_name, size_t expect_nbytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    EnsureCurrentDeviceContext(device_id);
    // must recheck under the lock scope
    if (module_[device_id] == nullptr) {
      CUDA_DRIVER_CALL(cuModuleLoadData(&(module_[device_id]), code_.data()));
      static auto nvshmem_init_hook = ffi::Function::GetGlobal("runtime.nvshmem.cumodule_init");
      if (nvshmem_init_hook.has_value()) {
        (*nvshmem_init_hook)(static_cast<void*>(module_[device_id]));
      }
    }
    CUdeviceptr global;
    size_t nbytes;
    CUresult result =
        cuModuleGetGlobal(&global, &nbytes, module_[device_id], global_name.c_str());
    if (result != CUDA_SUCCESS) {
      const char* msg;
      cuGetErrorName(result, &msg);
      TVM_FFI_THROW(CUDAError) << "cuModuleGetGlobal " << global_name
                               << " failed with error: " << msg;
    }
    TVM_FFI_ICHECK_EQ(nbytes, expect_nbytes);
    return global;
  }

  /*!
   * \brief JIT-compile raw CUDA C++ source to PTX/cubin/fatbin via the Python
   *        compile callback.  Called from BOTH the
   *        "ffi.Module.create.cuda" lambda (when the codegen hands us
   *        fmt=="cuda") AND from LoadFromBytes (when the saved-on-disk fmt is
   *        "cuda" — the cross-compile receiver path).
   *
   * \param source Raw CUDA C++ source (text).
   * \return Compiled binary bytes.  Determination of the compiled format
   *         (ptx vs cubin vs fatbin) is left to the caller (heuristic on
   *         first byte: '/' → ptx-text, otherwise binary).
   */
  static ffi::Bytes JitCompileFromSource(const ffi::String& source) {
    // Registry: "tvm_callback_cuda_compile" — Python-side nvcc/nvrtc wrapper.
    // Grep hint: grep -rn 'tvm_callback_cuda_compile' src/ python/
    auto fcompile = ffi::Function::GetGlobal("tvm_callback_cuda_compile");
    TVM_FFI_CHECK(fcompile.has_value(), RuntimeError)
        << "fmt=='cuda' requires tvm_callback_cuda_compile to be registered. "
        << "Import tvm.contrib.nvcc.";
    return (*fcompile)(source).cast<ffi::Bytes>();
  }

  /*! \brief Pick the compiled format from the JIT output's first byte. */
  static ffi::String DetermineCompiledFormat(const ffi::Bytes& compiled) {
    if (compiled.size() > 0 && compiled.data()[0] == '/') {
      return ffi::String("ptx");
    }
    return ffi::String("cubin");
  }

 private:
  // The binary data (compiled PTX/cubin/fatbin, or raw CUDA source if fmt == "cuda").
  ffi::Bytes code_;
  // The format of code_.
  ffi::String fmt_;
  // function information table.
  ffi::Map<ffi::String, FunctionInfo> fmap_;
  // In-memory source map for InspectSource — never serialized.
  ffi::Map<ffi::String, ffi::String> source_;
  // the internal modules per GPU, to be lazily initialized.
  std::array<CUmodule, kMaxNumGPUs> module_;
  // internal mutex when updating the module
  std::mutex mutex_;
};

// a wrapped function class to get packed func.
class CUDAWrappedFunc {
 public:
  // initialize the CUDA function.
  void Init(CUDAModuleNode* m, ffi::ObjectPtr<ffi::Object> sptr, const std::string& func_name,
            size_t num_void_args, const ffi::Array<ffi::String>& launch_param_tags) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    std::fill(fcache_.begin(), fcache_.end(), nullptr);
    // Track whether this kernel uses dynamic shared memory and the last size set per device.
    std::fill(dyn_smem_initialized_.begin(), dyn_smem_initialized_.end(), false);
    // Track whether cluster attribute has been set per device.
    std::fill(cluster_attr_initialized_.begin(), cluster_attr_initialized_.end(), false);
    use_dyn_shared_memory_ = false;
    for (const auto& tag : launch_param_tags) {
      if (tag == launch_param::kUseDynamicSharedMemoryTag) {
        use_dyn_shared_memory_ = true;
        break;
      }
    }
    launch_param_config_.Init(num_void_args, launch_param_tags);
  }
  // invoke the function with void arguments
  void operator()(ffi::PackedArgs args, ffi::Any* rv, void** void_args) const {
    int device_id;
    CUDA_CALL(cudaGetDevice(&device_id));
    EnsureCurrentDeviceContext(device_id);
    ThreadWorkLoad wl = launch_param_config_.Extract(args);

    if (fcache_[device_id] == nullptr) {
      fcache_[device_id] = m_->GetFunc(device_id, func_name_);
    }

    // If the kernel uses dynamic shared memory, we should ensure the attribute
    // reflects the actual size needed for this launch. Some workloads vary the
    // dynamic shared memory between invocations, in which case we cannot set it
    // just once. Cache the last value per device to avoid redundant calls.
    bool need_dyn_attr = use_dyn_shared_memory_ || (wl.dyn_shmem_size > 0);
    if (need_dyn_attr) {
      if (!dyn_smem_initialized_[device_id] || dyn_smem_last_[device_id] != wl.dyn_shmem_size) {
        CUresult attr_set = cuFuncSetAttribute(
            fcache_[device_id], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, wl.dyn_shmem_size);
        if (attr_set != CUDA_SUCCESS) {
          TVM_FFI_THROW(InternalError)
              << "Failed to set the allowed dynamic shared memory size to " << wl.dyn_shmem_size;
        }
        dyn_smem_last_[device_id] = wl.dyn_shmem_size;
        dyn_smem_initialized_[device_id] = true;
      }
    }
    CUstream strm = static_cast<CUstream>(TVMFFIEnvGetStream(kDLCUDA, device_id));
    CUresult result;

    TVM_FFI_ICHECK(wl.grid_dim(0) > 0 && wl.grid_dim(1) > 0 && wl.grid_dim(2) > 0)
        << "CUDALaunch Error: grid dimension must be positive, but got"
        << " grid=(" << wl.grid_dim(0) << "," << wl.grid_dim(1) << "," << wl.grid_dim(2) << ")"
        << " in kernel " << func_name_
        << ". A zero grid dimension is often caused by a dynamic shape"
        << " (e.g. num_tokens) being 0 at runtime.";

    if (wl.use_cluster_launch()) {
      // SM90+ cluster launch
      CUlaunchConfig config{};
      CUlaunchAttribute attribute[2]{};
      attribute[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      attribute[0].value.clusterDim.x = wl.cluster_dim[0];
      attribute[0].value.clusterDim.y = wl.cluster_dim[1];
      attribute[0].value.clusterDim.z = wl.cluster_dim[2];
      attribute[1].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
      attribute[1].value.programmaticStreamSerializationAllowed = 1;

      config.attrs = attribute;
      config.numAttrs = 2;
      config.hStream = strm;
      config.gridDimX = wl.grid_dim(0);
      config.gridDimY = wl.grid_dim(1);
      config.gridDimZ = wl.grid_dim(2);
      config.blockDimX = wl.block_dim(0);
      config.blockDimY = wl.block_dim(1);
      config.blockDimZ = wl.block_dim(2);
      config.sharedMemBytes = wl.dyn_shmem_size;

      // Set non-portable cluster size allowed attribute
      if (!cluster_attr_initialized_[device_id]) {
        CUresult attr_result = cuFuncSetAttribute(
            fcache_[device_id], CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1);
        if (attr_result != CUDA_SUCCESS) {
          const char* msg;
          cuGetErrorName(attr_result, &msg);
          TVM_FFI_THROW(InternalError) << "Failed to set cluster attribute for " << func_name_ << ": " << msg;
        }
        cluster_attr_initialized_[device_id] = true;
      }

      result = cuLaunchKernelEx(&config, fcache_[device_id], void_args, nullptr);
    } else if (launch_param_config_.use_programtic_dependent_launch()) {
      CUlaunchConfig config{};
      CUlaunchAttribute attribute[1]{};
      attribute[0].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
      attribute[0].value.programmaticStreamSerializationAllowed = 1;

      config.attrs = attribute;
      config.numAttrs = 1;
      config.hStream = strm;
      config.gridDimX = wl.grid_dim(0);
      config.gridDimY = wl.grid_dim(1);
      config.gridDimZ = wl.grid_dim(2);
      config.blockDimX = wl.block_dim(0);
      config.blockDimY = wl.block_dim(1);
      config.blockDimZ = wl.block_dim(2);
      config.sharedMemBytes = wl.dyn_shmem_size;

      result = cuLaunchKernelEx(&config, fcache_[device_id], void_args, nullptr);
    } else if (launch_param_config_.use_cooperative_launch()) {
      result = cuLaunchCooperativeKernel(fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1),
                                         wl.grid_dim(2), wl.block_dim(0), wl.block_dim(1),
                                         wl.block_dim(2), wl.dyn_shmem_size, strm, void_args);
    } else {
      result = cuLaunchKernel(fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1), wl.grid_dim(2),
                              wl.block_dim(0), wl.block_dim(1), wl.block_dim(2), wl.dyn_shmem_size,
                              strm, void_args, nullptr);
    }

    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {
      const char* msg;
      cuGetErrorName(result, &msg);
      std::ostringstream os;
      os << "CUDALaunch Error: " << msg << "\n"
         << " grid=(" << wl.grid_dim(0) << "," << wl.grid_dim(1) << "," << wl.grid_dim(2) << "), "
         << " block=(" << wl.block_dim(0) << "," << wl.block_dim(1) << "," << wl.block_dim(2)
         << ")"
         << " dyn_smem_bytes=" << wl.dyn_shmem_size << "\n";
      ffi::String cuda = m_->InspectSource("");
      if (cuda.length() != 0) {
        os << "// func_name=" << func_name_ << "\n"
           << "// CUDA Source\n"
           << "// -----------\n"
           << cuda;
      }
      TVM_FFI_THROW(InternalError) << os.str();
    }

    // Check for asynchronous CUDA errors that cuLaunchKernel's return value
    // does not capture (e.g. illegal memory access during kernel execution).
    // This matches the Cython backend's TILELANG_CHECK_LAST_ERROR macro.
    if (result == CUDA_SUCCESS) {
      cudaError_t last_err = cudaPeekAtLastError();
      if (last_err != cudaSuccess) {
        // Use driver API cuGetErrorName for the error name (cudaGetErrorName
        // is not available in the cudart stub). The numeric values of
        // cudaError_t and CUresult are identical for matching error codes.
        const char* err_name = nullptr;
        cuGetErrorName(static_cast<CUresult>(last_err), &err_name);
        const char* err_str = cudaGetErrorString(last_err);
        // Clear the sticky error so subsequent CUDA calls are not poisoned.
        cudaGetLastError();
        TVM_FFI_THROW(InternalError) << func_name_ << ": " << (err_name ? err_name : "unknown") << " - " << err_str;
      }
    }
  }

 private:
  // internal module
  CUDAModuleNode* m_;
  // the resource holder
  ffi::ObjectPtr<ffi::Object> sptr_;
  // The name of the function.
  std::string func_name_;
  // Device function cache per device.
  // mark as mutable, to enable lazy initialization
  mutable std::array<CUfunction, kMaxNumGPUs> fcache_;
  // launch parameters configuration
  LaunchParamConfig launch_param_config_;
  // Whether this kernel uses dynamic shared memory
  bool use_dyn_shared_memory_{false};
  // Cached last dynamic shared memory size per device and whether it's initialized
  mutable std::array<size_t, kMaxNumGPUs> dyn_smem_last_;
  mutable std::array<bool, kMaxNumGPUs> dyn_smem_initialized_;
  // Whether cluster attribute has been initialized per device
  mutable std::array<bool, kMaxNumGPUs> cluster_attr_initialized_;
  // have pdl setting
  bool has_programmatic_dependent_launch_;
};



ffi::Optional<ffi::Function> CUDAModuleNode::GetFunction(const ffi::String& name) {
  ffi::ObjectPtr<ffi::Object> sptr_to_self = ffi::GetObjectPtr<ffi::Object>(this);
  TVM_FFI_ICHECK_EQ(sptr_to_self.get(), this);
  auto opt_info = fmap_.Get(name);
  if (!opt_info.has_value()) return ffi::Function();
  FunctionInfo info = opt_info.value();
  CUDAWrappedFunc f;
  f.Init(this, sptr_to_self, name, info->arg_types.size(), info->launch_param_tags);
  return PackFuncVoidAddr(f, info->arg_types, info->arg_extra_tags);
}

// Construct a CUDAModuleNode from in-memory payload.  When fmt == "cuda" the
// code is raw CUDA C++ source — JIT-compile via the Python callback, then
// re-tag with the resulting compiled format ("ptx" / "cubin").
static ffi::Module CUDAModuleCreateImpl(ffi::Bytes code, ffi::String fmt,
                                        ffi::Map<ffi::String, FunctionInfo> fmap,
                                        ffi::Map<ffi::String, ffi::String> source) {
  if (fmt == "cuda") {
    // Stash the CUDA source for InspectSource before we replace `code` with
    // the JIT output.
    if (source.find("cuda") == source.end()) {
      source.Set("cuda", ffi::String(code.data(), code.size()));
    }
    ffi::Bytes compiled =
        CUDAModuleNode::JitCompileFromSource(ffi::String(code.data(), code.size()));
    fmt = CUDAModuleNode::DetermineCompiledFormat(compiled);
    code = std::move(compiled);
  }
  auto n = ffi::make_object<CUDAModuleNode>(code, fmt, fmap, source);
  return ffi::Module(n);
}

static ffi::Module CUDAModuleLoadFromBytes(const ffi::Bytes& bytes) {
  support::BytesInStream stream(bytes);
  ffi::String fmt;
  ffi::Map<ffi::String, FunctionInfo> fmap;
  ffi::Bytes code;
  stream.Read(&fmt);
  TVM_FFI_ICHECK(stream.Read(&fmap));
  stream.Read(&code);
  // Source map is not serialized — it is lost on save/load round-trip.
  // If the receiver wants InspectSource("cuda") to work, the saved bytes must
  // have been written with fmt=="cuda" so the JIT path below re-stuffs the
  // source map with the original C++ source.
  return CUDAModuleCreateImpl(std::move(code), std::move(fmt), std::move(fmap),
                              ffi::Map<ffi::String, ffi::String>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // Registry: "ffi.Module.create.cuda" — codegen-time CUDA module factory.
  // Used by src/target/cuda/cuda_fallback_module.h:CUDAModuleCreateWithFallback.
  // Registry: "ffi.Module.load_from_bytes.cuda" — disk loader.  Only this
  // (real) module registers a loader; the fallback module is codegen-time only.
  refl::GlobalDef()
      .def("ffi.Module.load_from_bytes.cuda", CUDAModuleLoadFromBytes)
      .def("ffi.Module.create.cuda",
           [](ffi::Bytes code, ffi::String fmt, ffi::Map<ffi::String, FunctionInfo> fmap,
              ffi::Map<ffi::String, ffi::String> source) {
             return CUDAModuleCreateImpl(std::move(code), std::move(fmt), std::move(fmap),
                                         std::move(source));
           });
}
}  // namespace runtime
}  // namespace tvm
