import torch
import tvm.tl.language as T

from tvm.tl.engine import compile
from tvm.tl.utils import ConvertTorch, TensorSupplyType
from functools import partial

def convolution(N, C, H, W, INC, KW, KH, P, S, D):
    INH = (H - 1) * S + (KH - 1) * D + 1 - 2 * P
    INW = (W - 1) * S + (KW - 1) * D + 1 - 2 * P

    dtype = "float16"
    block_M = 128
    block_N = 128
    block_K = 32

    @T.prim_func
    def main(data: T.Buffer((N, INH, INW, INC), dtype), kernel: T.Buffer((KH * KW * INC, C), dtype),
             out: T.Buffer((N * H * W, C), dtype)):

        bx, by, _ = T.launch_program(T.ceildiv(C, block_N), T.ceildiv(N*H*W, block_M), num_threads=128)

        with T.block():
            data_shared = T.alloc_buffer((block_M, block_K), dtype, scope="shared.dyn")
            kernel_shared = T.alloc_buffer((block_K, block_N), dtype, scope="shared.dyn")
            out_local = T.alloc_buffer((block_M, block_N), "float", scope="local.fragment")
            T.clear(out_local)

            for k_iter in T.Pipelined(T.ceildiv(KH*KW*INC, block_K), num_stages=3):
                for i, j in T.Parallel(block_M, block_K):
                    k = k_iter * block_K + j
                    m = by * block_M + i
                    access_h = m % (H*W) // W * S + k // (KW * INC) * D - P
                    access_w = m % W * S + k // INC % KW * D - P
                    in_bound = (access_h >= 0) and (access_w >= 0) and (access_h < INH) and (access_w < INW)
                    data_shared[i, j] = T.if_then_else(in_bound, data[m//(H*W), access_h, access_w, k%INC], 0)
                T.copy(kernel[k_iter*block_K, bx*block_N], kernel_shared)
                T.gemm(data_shared, kernel_shared, out_local)

            T.copy(out_local, out[by*block_M, bx*block_N])

    return main

def conv_nhwc_ref(A: torch.Tensor, B: torch.Tensor, stride, padding, dilation):
    A = A.permute(0, 3, 1, 2) # N, H, W, C -> N, C, H, W
    B = B.permute(3, 2, 0, 1) # H, W, C, F -> F, C, H, W
    C = torch.conv2d(A, B, stride=stride, padding=padding, dilation=dilation)
    C = C.permute(0, 2, 3, 1) # N, C, H, W -> N, H, W, C
    return C

def ref_program(A, B, stride, padding, dilation):
    B = B.reshape(3, 3, A.shape[-1], -1)
    C = conv_nhwc_ref(A, B, stride, padding, dilation)
    C = C.reshape(-1, C.shape[-1])
    return [C]

if __name__ == "__main__":
    N, C, H, W, INC, KW, KH, P, S, D = 8, 256, 128, 128, 256, 3, 3, 1, 1, 1
    program = convolution(N, C, H, W, INC, KW, KH, P, S, D)
    ref_program = partial(ref_program, stride=S, padding=P, dilation=D)
    mod, params = compile(program)
    supply_type = TensorSupplyType.Integer
    mod = ConvertTorch(mod, params, [2], supply_type)
    mod.assert_allclose(ref_program)

    total_flops = 2 * N * C * H * W * INC * KH * KW

    latency = mod.do_bench(ref_program, warmup=500)
    print("{:.2f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = mod.do_bench(mod)
    print("{:.2f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
