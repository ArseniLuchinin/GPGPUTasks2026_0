#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void matrix_multiply_via_local_memory(
    const float* a, // rows=h x cols=k
    const float* b, // rows=k x cols=w
    float* c, // rows=h x cols=w
    unsigned int w,
    unsigned int h,
    unsigned int k)
{
    __shared__ float as[GROUP_SIZE_Y][GROUP_SIZE_X];
    __shared__ float bs[GROUP_SIZE_Y][GROUP_SIZE_X];

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;

    const unsigned int row = blockIdx.y * GROUP_SIZE_Y + ty;
    const unsigned int col = blockIdx.x * GROUP_SIZE_X + tx;

    float acc = 0.0f;

    for (unsigned int kk = 0; kk < k; kk += GROUP_SIZE_X) {
        const unsigned int a_col = kk + tx;
        as[ty][tx] = (row < h && a_col < k) ? a[row * k + a_col] : 0.0f;

        const unsigned int b_row = kk + ty;
        bs[ty][tx] = (b_row < k && col < w) ? b[b_row * w + col] : 0.0f;

        __syncthreads();

#pragma unroll
        for (unsigned int t = 0; t < GROUP_SIZE_X; ++t)
            acc += as[ty][t] * bs[t][tx];

        __syncthreads();
    }

    if (row < h && col < w)
        c[row * w + col] = acc;
}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, gpu::gpu_mem_32f& c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
