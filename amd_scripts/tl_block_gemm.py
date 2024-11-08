# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tvm
print(tvm.__path__)
from tvm import tl

@tvm.register_func("tvm_callback_hip_postproc", override=True)
def tvm_callback_hip_postproc(code, _):
    # code = code.replace("""#pragma unroll
    # for (int i_1 = 0; i_1 < 2; ++i_1) {
    #   *(uint4*)(A_shared + ((i_1 * 2048) + (((int)threadIdx.x) * 8))) = *(uint4*)(A + (((((((int)blockIdx.y) * 2097152) + (i_1 * 1048576)) + ((((int)threadIdx.x) >> 2) * 16384)) + (k * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    # }""", r"""#pragma unroll
    # for (int i_1 = 0; i_1 < 2; ++i_1) {
    #   *(uint4*)(A_shared + ((i_1 * 2048) + (((int)threadIdx.x) * 8))) = *(uint4*)(A + ((((((((int)blockIdx.y) * 2097152) + (i_1 * 1048576)) + (((int)((threadIdx.x / 64) % 2)) * 524288)) + (((int)((threadIdx.x / 64) / 2)) * 262144)) + (k * 512)) + (((int)threadIdx.x % 64) * 8)));
    # }""")
    # code = code.replace("""#pragma unroll
    # for (int i_2 = 0; i_2 < 2; ++i_2) {
    #   *(uint4*)(B_shared + ((i_2 * 2048) + (((int)threadIdx.x) * 8))) = *(uint4*)(B + (((((((int)blockIdx.x) * 2097152) + (i_2 * 1048576)) + ((((int)threadIdx.x) >> 2) * 16384)) + (k * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    # }""", r"""#pragma unroll
    #     for (int i_2 = 0; i_2 < 2; ++i_2) {
    #   *(uint4*)(B_shared + ((((i_2 * 2048) + (((int)((threadIdx.x / 64) % 2)) * 1024)) + (((int)(threadIdx.x / 128)) * 512)) + (((int)(threadIdx.x % 64)) * 8))) = *(uint4*)(B + ((((((((int)blockIdx.x) * 2097152) + (i_2 * 1048576)) + (((int)((threadIdx.x / 64) % 2)) * 524288)) + (((int)threadIdx.z) * 262144)) + (k * 512)) + (((int)(threadIdx.x % 64)) * 8)));
    # }""")
    # print(code)
    return code

def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    dtypeAB,
    dtypeC,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    import tvm.tl.language as T

    @T.prim_func
    def main(
        A: T.Buffer(A_shape, dtypeAB),
        B: T.Buffer(B_shape, dtypeAB),
        C: T.Buffer((M, N), dtypeC),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, dtypeAB, scope="shared")
            B_shared = T.alloc_shared(B_shared_shape, dtypeAB, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            
            # T.use_swizzle(10)

            T.clear(C_local)
            for k in T.serial(T.ceildiv(K, block_K)):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, False, True)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm(
    M,
    N,
    K,
    trans_A,
    trans_B,
    dtypeAB,
    dtypeC,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=3,
    num_threads=128,
):
    program = matmul(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        dtypeAB,
        dtypeC,
        dtypeAccum,
        num_stages,
        num_threads,
    )
    mod, params = tl.lower(program, target="hip")
    # print(mod.imported_modules[0].get_source())
    mod = tl.Profiler(mod, params, [2], tl.TensorSupplyType.Integer)
    import torch
    # a = torch.randn((M, K), dtype=torch.__getattribute__(dtypeAB)).to("cuda")
    # b = torch.randn((N, K), dtype=torch.__getattribute__(dtypeAB)).to("cuda")
    a = torch.ones((M, K), dtype=torch.__getattribute__(dtypeAB)).to("cuda")
    b = torch.ones((N, K), dtype=torch.__getattribute__(dtypeAB)).to("cuda")
    c = mod(a, b)

    print(c)

    ref_c = torch.matmul(a, b.T).to(torch.__getattribute__(dtypeC))
    print(ref_c)

    latency = mod.do_bench(mod.func, profiler="tvm")
    print(f"Latency: {latency}")

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    

if __name__ == "__main__":
    run_gemm(
        16384,
        16384,
        16384,
        False,
        True,
        "float16",
        "float32",
        "float32",
        128,
        128,
        32,
        2,
        256,
    )
