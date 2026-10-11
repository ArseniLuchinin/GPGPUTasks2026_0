#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void matrix_transpose_coalesced_via_local_memory(
    const float* matrix, // w x h
    float* transposed_matrix, // h x w
    unsigned int w,
    unsigned int h)
{

    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= w || row >= h)
        return;

    // TODO
    __shared__ float table[GROUP_SIZE_Y][GROUP_SIZE_X + 1];
    table[threadIdx.y][threadIdx.x] = matrix[row * w + col];
    __syncthreads();

    const unsigned int inRow = blockIdx.y * blockDim.y + threadIdx.x; // строка входа
    const unsigned int inCol = blockIdx.x * blockDim.x + threadIdx.y; // столбец входа

    transposed_matrix[inCol * h + inRow] = table[threadIdx.x][threadIdx.y];
}

namespace cuda {
void matrix_transpose_coalesced_via_local_memory(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& matrix, gpu::gpu_mem_32f& transposed_matrix, unsigned int w, unsigned int h)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_transpose_coalesced_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(matrix.cuptr(), transposed_matrix.cuptr(), w, h);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
