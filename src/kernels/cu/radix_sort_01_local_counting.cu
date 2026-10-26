#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Общие константы процедуры radix sort ( зеркалятся в radix_sort_04_scatter.cu ):
// один разряд = один бит (примерено 0..31), за один запуск обрабатывается один бит;
// тайл - кусок массива, обрабатываемый одним блоком: GROUP_SIZE нитей по RADIX_ITEMS_PER_THREAD элементов
#define RADIX_ITEMS_PER_THREAD 16u
#define RADIX_TILE (GROUP_SIZE * RADIX_ITEMS_PER_THREAD)

__global__ void radix_sort_01_local_counting(
    const unsigned int* input,
          unsigned int* zeros_counts, // для каждого тайла - количество элементов с нулём в текущем разряде
    unsigned int n,
    unsigned int digit) // номер бита (0..31)
{
    const unsigned int base = blockIdx.x * RADIX_TILE + threadIdx.x * RADIX_ITEMS_PER_THREAD;

    unsigned int zeros = 0;
    for (unsigned int k = 0; k < RADIX_ITEMS_PER_THREAD; ++k) {
        const unsigned int idx = base + k;
        if (idx < n) {
            const unsigned int bit = (input[idx] >> digit) & 1u;
            zeros += (bit ^ 1u);
        }
    }

    // блок-скан - чтобы посчитать суммарное число нулей в тайле
    __shared__ unsigned int s[GROUP_SIZE];
    s[threadIdx.x] = zeros;
    __syncthreads();
    for (unsigned int stride = 1; stride < GROUP_SIZE; stride <<= 1) {
        unsigned int add = 0;
        __syncthreads();
        if (threadIdx.x >= stride) {
            add = s[threadIdx.x - stride];
        }
        __syncthreads();
        s[threadIdx.x] += add;
    }
    __syncthreads();

    zeros_counts[blockIdx.x] = s[GROUP_SIZE - 1];
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input, gpu::gpu_mem_32u &zeros_counts, unsigned int n, unsigned int digit)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input.cuptr(), zeros_counts.cuptr(), n, digit);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
