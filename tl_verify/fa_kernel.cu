#include <tl_templates/gemm_sm90.h>
#include <tl_templates/copy.h>
#include <tl_templates/copy_sm90.h>
#include <tl_templates/reduce.h>
#include <tl_templates/ldsm.h>
#include <tl_templates/threadblock_swizzle.h>
#include "fa_kernel.hpp"

extern "C" __global__ void __launch_bounds__(128) main_kernel(__grid_constant__ const CUtensorMap K_desc, half_t* __restrict__ Output, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc) {
}

template <typename T>
static std::string ArrayToStr(const T* ptr, size_t n) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < n; i++) {
    if (i > 0) ss << ", ";
    ss << ptr[i];
  }
  ss << "]";
  return ss.str();
}

struct TensorMapArgs {
  CUtensorMap* map;
  CUtensorMapDataType type;
  cuuint32_t tensorRank;
  void* globalAddress;
  cuuint64_t globalDim[5], globalStride[5];
  cuuint32_t boxDim[5], elementStrides[5];
  CUtensorMapInterleave interleave;
  CUtensorMapSwizzle swizzle;
  CUtensorMapL2promotion l2Promotion;
  CUtensorMapFloatOOBfill oobFill;

  std::string ToDebugString() {
    std::stringstream ss;
    ss << "TMA Desc Addr:   " << map << std::endl
       << "format         " << type << std::endl
       << "dim            " << tensorRank << std::endl
       << "gmem_address   " << globalAddress << std::endl
       << "globalDim      " << ArrayToStr(globalDim, tensorRank) << std::endl
       << "globalStrides  " << ArrayToStr(globalStride, tensorRank) << std::endl
       << "boxDim         " << ArrayToStr(boxDim, tensorRank) << std::endl
       << "elementStrides " << ArrayToStr(elementStrides, tensorRank) << std::endl
       << "interleave     " << interleave << std::endl
       << "swizzle        " << swizzle << std::endl
       << "l2Promotion    " << l2Promotion << std::endl
       << "oobFill        " << oobFill << std::endl;
    return ss.str();
  }
};

void host_function(Flash_fwd_params params) {
  int num_m_blocks = (params.seq_len + params.block_M - 1) / params.block_M;
  dim3 grid(num_m_blocks, params.head, params.batch);
  dim3 block(128);
  size_t sharedMemSize = (params.block_M + 2 * params.block_N) * params.dim * sizeof(half_t); // 24576;

  // int size = params.batch * params.head * params.seq_len * params.dim * sizeof(half_t);

  CUtensorMap Q_desc = {0};
  CUtensorMap K_desc = {0};
  CUtensorMap V_desc = {0};
  TensorMapArgs Q_arg;
  TensorMapArgs K_arg;
  TensorMapArgs V_arg;

  Q_arg.map = &Q_desc;
  Q_arg.type = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  Q_arg.tensorRank = 4;
  Q_arg.globalAddress = params.q_ptr;
  Q_arg.globalDim[0] = static_cast<cuuint64_t>(params.dim);
  Q_arg.globalDim[1] = static_cast<cuuint64_t>(params.head);
  Q_arg.globalDim[2] = static_cast<cuuint64_t>(params.seq_len);
  Q_arg.globalDim[3] = static_cast<cuuint64_t>(params.batch);
  Q_arg.globalStride[0] = static_cast<cuuint64_t>(2);
  Q_arg.globalStride[1] = static_cast<cuuint64_t>(128);
  Q_arg.globalStride[2] = static_cast<cuuint64_t>(128);
  Q_arg.globalStride[3] = static_cast<cuuint64_t>(32768);
  Q_arg.boxDim[0] = static_cast<cuuint64_t>(64);
  Q_arg.boxDim[1] = static_cast<cuuint64_t>(1);
  Q_arg.boxDim[2] = static_cast<cuuint64_t>(64);
  Q_arg.boxDim[3] = static_cast<cuuint64_t>(1);
  Q_arg.elementStrides[0] = static_cast<cuuint64_t>(1);
  Q_arg.elementStrides[1] = static_cast<cuuint64_t>(1);
  Q_arg.elementStrides[2] = static_cast<cuuint64_t>(1);
  Q_arg.elementStrides[3] = static_cast<cuuint64_t>(1);
  Q_arg.interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
  Q_arg.swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
  Q_arg.l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
  Q_arg.oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

  K_arg.map = &K_desc;
  K_arg.type = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  K_arg.tensorRank = 4;
  K_arg.globalAddress = params.k_ptr;
  K_arg.globalDim[0] = static_cast<cuuint64_t>(64);
  K_arg.globalDim[1] = static_cast<cuuint64_t>(1);
  K_arg.globalDim[2] = static_cast<cuuint64_t>(256);
  K_arg.globalDim[3] = static_cast<cuuint64_t>(1);
  K_arg.globalStride[0] = static_cast<cuuint64_t>(2);
  K_arg.globalStride[1] = static_cast<cuuint64_t>(128);
  K_arg.globalStride[2] = static_cast<cuuint64_t>(128);
  K_arg.globalStride[3] = static_cast<cuuint64_t>(32768);
  K_arg.boxDim[0] = static_cast<cuuint64_t>(64);
  K_arg.boxDim[1] = static_cast<cuuint64_t>(1);
  K_arg.boxDim[2] = static_cast<cuuint64_t>(64);
  K_arg.boxDim[3] = static_cast<cuuint64_t>(1);
  K_arg.elementStrides[0] = static_cast<cuuint64_t>(1);
  K_arg.elementStrides[1] = static_cast<cuuint64_t>(1);
  K_arg.elementStrides[2] = static_cast<cuuint64_t>(1);
  K_arg.elementStrides[3] = static_cast<cuuint64_t>(1);
  K_arg.interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
  K_arg.swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
  K_arg.l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
  K_arg.oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

  V_arg.map = &V_desc;
  V_arg.type = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  V_arg.tensorRank = 4;
  V_arg.globalAddress = params.v_ptr;
  V_arg.globalDim[0] = static_cast<cuuint64_t>(64);
  V_arg.globalDim[1] = static_cast<cuuint64_t>(1);
  V_arg.globalDim[2] = static_cast<cuuint64_t>(256);
  V_arg.globalDim[3] = static_cast<cuuint64_t>(1);
  V_arg.globalStride[0] = static_cast<cuuint64_t>(2);
  V_arg.globalStride[1] = static_cast<cuuint64_t>(128);
  V_arg.globalStride[2] = static_cast<cuuint64_t>(128);
  V_arg.globalStride[3] = static_cast<cuuint64_t>(32768);
  V_arg.boxDim[0] = static_cast<cuuint64_t>(64);
  V_arg.boxDim[1] = static_cast<cuuint64_t>(1);
  V_arg.boxDim[2] = static_cast<cuuint64_t>(64);
  V_arg.boxDim[3] = static_cast<cuuint64_t>(1);
  V_arg.elementStrides[0] = static_cast<cuuint64_t>(1);
  V_arg.elementStrides[1] = static_cast<cuuint64_t>(1);
  V_arg.elementStrides[2] = static_cast<cuuint64_t>(1);
  V_arg.elementStrides[3] = static_cast<cuuint64_t>(1);
  V_arg.interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
  V_arg.swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
  V_arg.l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
  V_arg.oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

  CUresult result;
  result = cuTensorMapEncodeTiled(
      Q_arg.map, Q_arg.type, Q_arg.tensorRank, Q_arg.globalAddress, Q_arg.globalDim, Q_arg.globalStride + 1, Q_arg.boxDim,
      Q_arg.elementStrides, Q_arg.interleave, Q_arg.swizzle, Q_arg.l2Promotion, Q_arg.oobFill);
  if (result != CUDA_SUCCESS) {
    std::cout << "Failed to initialize the TMA descriptor " << result << std::endl
              << Q_arg.ToDebugString();
  }

  result = cuTensorMapEncodeTiled(
      K_arg.map, K_arg.type, K_arg.tensorRank, K_arg.globalAddress, K_arg.globalDim, K_arg.globalStride + 1, K_arg.boxDim,
      K_arg.elementStrides, K_arg.interleave, K_arg.swizzle, K_arg.l2Promotion, K_arg.oobFill);
  if (result != CUDA_SUCCESS) {
    std::cout << "Failed to initialize the TMA descriptor " << result << std::endl
              << K_arg.ToDebugString();
  }

  result = cuTensorMapEncodeTiled(
      V_arg.map, V_arg.type, V_arg.tensorRank, V_arg.globalAddress, V_arg.globalDim, V_arg.globalStride + 1, V_arg.boxDim,
      V_arg.elementStrides, V_arg.interleave, V_arg.swizzle, V_arg.l2Promotion, V_arg.oobFill);
  if (result != CUDA_SUCCESS) {
    std::cout << "Failed to initialize the TMA descriptor " << result << std::endl
              << V_arg.ToDebugString();
  }

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
      std::cerr << "CUDA device synchronization failed: " << cudaGetErrorString(err) << std::endl;
      return;
  }

  main_kernel<<<grid, block, sharedMemSize>>>(K_desc, (half_t*)params.output_ptr, Q_desc, V_desc);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
      std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
      return;
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
      std::cerr << "CUDA device synchronization failed: " << cudaGetErrorString(err) << std::endl;
      return;
  }
}
// template <typename T>
// static std::string ArrayToStr(const T* ptr, size_t n) {
//   std::stringstream ss;
//   ss << "[";
//   for (size_t i = 0; i < n; i++) {
//     if (i > 0) ss << ", ";
//     ss << ptr[i];
//   }
//   ss << "]";
//   return ss.str();
// }

// struct TensorMapArgs {
//   CUtensorMap* map;
//   CUtensorMapDataType type;
//   cuuint32_t tensorRank;
//   void* globalAddress;
//   cuuint64_t globalDim[5], globalStride[5];
//   cuuint32_t boxDim[5], elementStrides[5];
//   CUtensorMapInterleave interleave;
//   CUtensorMapSwizzle swizzle;
//   CUtensorMapL2promotion l2Promotion;
//   CUtensorMapFloatOOBfill oobFill;

//   std::string ToDebugString() {
//     std::stringstream ss;
//     ss << "TMA Desc Addr:   " << map << std::endl
//        << "format         " << type << std::endl
//        << "dim            " << tensorRank << std::endl
//        << "gmem_address   " << globalAddress << std::endl
//        << "globalDim      " << ArrayToStr(globalDim, tensorRank) << std::endl
//        << "globalStrides  " << ArrayToStr(globalStride, tensorRank) << std::endl
//        << "boxDim         " << ArrayToStr(boxDim, tensorRank) << std::endl
//        << "elementStrides " << ArrayToStr(elementStrides, tensorRank) << std::endl
//        << "interleave     " << interleave << std::endl
//        << "swizzle        " << swizzle << std::endl
//        << "l2Promotion    " << l2Promotion << std::endl
//        << "oobFill        " << oobFill << std::endl;
//     return ss.str();
//   }
// };

// __global__ void fillWithOnes(void *ptr, size_t size) {
//   half_t *data = (half_t *)ptr;
//   size_t index = threadIdx.x + blockIdx.x * blockDim.x;
//   if (index < size) {
//     data[index] = 1;
//   }
// }

// int main() {
//   dim3 grid(4);
//   dim3 block(129);
//   size_t sharedMemSize = 24576;

//   int batch = 1;
//   int head = 1;
//   int seq_len = 256;
//   int dim = 64;
//   int size = batch * head * seq_len * dim * sizeof(half_t);

//   void *Q, *K, *V, *d_output;
//   void *h_output;
//   h_output = (void*)malloc(size);
//   cudaMalloc((void**)&Q, size);
//   cudaMalloc((void**)&K, size);
//   cudaMalloc((void**)&V, size);
//   cudaMalloc((void**)&d_output, size);


//   int threadsPerBlock = 256;
//   int blocksPerGrid = (batch * head * seq_len * dim + threadsPerBlock - 1) / threadsPerBlock;
//   fillWithOnes<<<blocksPerGrid, threadsPerBlock>>>(Q, batch * head * seq_len * dim + threadsPerBlock);
//   fillWithOnes<<<blocksPerGrid, threadsPerBlock>>>(K, batch * head * seq_len * dim + threadsPerBlock);
//   fillWithOnes<<<blocksPerGrid, threadsPerBlock>>>(V, batch * head * seq_len * dim + threadsPerBlock);

//   cudaError_t err = cudaDeviceSynchronize();
//   if (err != cudaSuccess) {
//       std::cerr << "fillWithOnes failed: " << cudaGetErrorString(err) << std::endl;
//       return 1;
//   }

//   CUtensorMap Q_desc = {0};
//   CUtensorMap K_desc = {0};
//   CUtensorMap V_desc = {0};
//   TensorMapArgs Q_arg;
//   TensorMapArgs K_arg;
//   TensorMapArgs V_arg;

//   Q_arg.map = &Q_desc;
//   Q_arg.type = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
//   Q_arg.tensorRank = 4;
//   Q_arg.globalAddress = Q;
//   Q_arg.globalDim[0] = static_cast<cuuint64_t>(64);
//   Q_arg.globalDim[1] = static_cast<cuuint64_t>(1);
//   Q_arg.globalDim[2] = static_cast<cuuint64_t>(256);
//   Q_arg.globalDim[3] = static_cast<cuuint64_t>(1);
//   Q_arg.globalStride[0] = static_cast<cuuint64_t>(2);
//   Q_arg.globalStride[1] = static_cast<cuuint64_t>(128);
//   Q_arg.globalStride[2] = static_cast<cuuint64_t>(128);
//   Q_arg.globalStride[3] = static_cast<cuuint64_t>(32768);
//   Q_arg.boxDim[0] = static_cast<cuuint64_t>(64);
//   Q_arg.boxDim[1] = static_cast<cuuint64_t>(1);
//   Q_arg.boxDim[2] = static_cast<cuuint64_t>(64);
//   Q_arg.boxDim[3] = static_cast<cuuint64_t>(1);
//   Q_arg.elementStrides[0] = static_cast<cuuint64_t>(1);
//   Q_arg.elementStrides[1] = static_cast<cuuint64_t>(1);
//   Q_arg.elementStrides[2] = static_cast<cuuint64_t>(1);
//   Q_arg.elementStrides[3] = static_cast<cuuint64_t>(1);
//   Q_arg.interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
//   Q_arg.swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
//   Q_arg.l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
//   Q_arg.oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

//   K_arg.map = &K_desc;
//   K_arg.type = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
//   K_arg.tensorRank = 4;
//   K_arg.globalAddress = K;
//   K_arg.globalDim[0] = static_cast<cuuint64_t>(64);
//   K_arg.globalDim[1] = static_cast<cuuint64_t>(1);
//   K_arg.globalDim[2] = static_cast<cuuint64_t>(256);
//   K_arg.globalDim[3] = static_cast<cuuint64_t>(1);
//   K_arg.globalStride[0] = static_cast<cuuint64_t>(2);
//   K_arg.globalStride[1] = static_cast<cuuint64_t>(128);
//   K_arg.globalStride[2] = static_cast<cuuint64_t>(128);
//   K_arg.globalStride[3] = static_cast<cuuint64_t>(32768);
//   K_arg.boxDim[0] = static_cast<cuuint64_t>(64);
//   K_arg.boxDim[1] = static_cast<cuuint64_t>(1);
//   K_arg.boxDim[2] = static_cast<cuuint64_t>(64);
//   K_arg.boxDim[3] = static_cast<cuuint64_t>(1);
//   K_arg.elementStrides[0] = static_cast<cuuint64_t>(1);
//   K_arg.elementStrides[1] = static_cast<cuuint64_t>(1);
//   K_arg.elementStrides[2] = static_cast<cuuint64_t>(1);
//   K_arg.elementStrides[3] = static_cast<cuuint64_t>(1);
//   K_arg.interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
//   K_arg.swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
//   K_arg.l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
//   K_arg.oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

//   V_arg.map = &V_desc;
//   V_arg.type = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
//   V_arg.tensorRank = 4;
//   V_arg.globalAddress = V;
//   V_arg.globalDim[0] = static_cast<cuuint64_t>(64);
//   V_arg.globalDim[1] = static_cast<cuuint64_t>(1);
//   V_arg.globalDim[2] = static_cast<cuuint64_t>(256);
//   V_arg.globalDim[3] = static_cast<cuuint64_t>(1);
//   V_arg.globalStride[0] = static_cast<cuuint64_t>(2);
//   V_arg.globalStride[1] = static_cast<cuuint64_t>(128);
//   V_arg.globalStride[2] = static_cast<cuuint64_t>(128);
//   V_arg.globalStride[3] = static_cast<cuuint64_t>(32768);
//   V_arg.boxDim[0] = static_cast<cuuint64_t>(64);
//   V_arg.boxDim[1] = static_cast<cuuint64_t>(1);
//   V_arg.boxDim[2] = static_cast<cuuint64_t>(64);
//   V_arg.boxDim[3] = static_cast<cuuint64_t>(1);
//   V_arg.elementStrides[0] = static_cast<cuuint64_t>(1);
//   V_arg.elementStrides[1] = static_cast<cuuint64_t>(1);
//   V_arg.elementStrides[2] = static_cast<cuuint64_t>(1);
//   V_arg.elementStrides[3] = static_cast<cuuint64_t>(1);
//   V_arg.interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
//   V_arg.swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
//   V_arg.l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
//   V_arg.oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

//   CUresult result;
//   result = cuTensorMapEncodeTiled(
//       Q_arg.map, Q_arg.type, Q_arg.tensorRank, Q_arg.globalAddress, Q_arg.globalDim, Q_arg.globalStride + 1, Q_arg.boxDim,
//       Q_arg.elementStrides, Q_arg.interleave, Q_arg.swizzle, Q_arg.l2Promotion, Q_arg.oobFill);
//   if (result != CUDA_SUCCESS) {
//     std::cout << "Failed to initialize the TMA descriptor " << result << std::endl
//               << Q_arg.ToDebugString();
//   }

//   result = cuTensorMapEncodeTiled(
//       K_arg.map, K_arg.type, K_arg.tensorRank, K_arg.globalAddress, K_arg.globalDim, K_arg.globalStride + 1, K_arg.boxDim,
//       K_arg.elementStrides, K_arg.interleave, K_arg.swizzle, K_arg.l2Promotion, K_arg.oobFill);
//   if (result != CUDA_SUCCESS) {
//     std::cout << "Failed to initialize the TMA descriptor " << result << std::endl
//               << K_arg.ToDebugString();
//   }

//   result = cuTensorMapEncodeTiled(
//       V_arg.map, V_arg.type, V_arg.tensorRank, V_arg.globalAddress, V_arg.globalDim, V_arg.globalStride + 1, V_arg.boxDim,
//       V_arg.elementStrides, V_arg.interleave, V_arg.swizzle, V_arg.l2Promotion, V_arg.oobFill);
//   if (result != CUDA_SUCCESS) {
//     std::cout << "Failed to initialize the TMA descriptor " << result << std::endl
//               << V_arg.ToDebugString();
//   }

//   if (err != cudaSuccess) {
//       std::cerr << "CUDA device synchronization failed: " << cudaGetErrorString(err) << std::endl;
//       return 1;
//   }

//   main_kernel<<<grid, block, sharedMemSize>>>(K_desc, (half_t*)d_output, Q_desc, V_desc);

//   err = cudaGetLastError();
//   if (err != cudaSuccess) {
//       std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
//       return 1;
//   }

//   err = cudaDeviceSynchronize();
//   if (err != cudaSuccess) {
//       std::cerr << "CUDA device synchronization failed: " << cudaGetErrorString(err) << std::endl;
//       return 1;
//   }

//   cudaMemcpy((void*)h_output, (void*)d_output, size, cudaMemcpyDeviceToHost);

//   std::cout << "CUDA kernel executed successfully." << std::endl;
//   for (int i = 0; i < seq_len; i++) {
//     for (int j = 0; j < dim; j++) {
//       std::cout << ((half_t*)h_output)[i * dim + j] << " ";
//     }
//     std::cout << std::endl;
//   }
//   std::cout << std::endl;
//   return 0;
// }