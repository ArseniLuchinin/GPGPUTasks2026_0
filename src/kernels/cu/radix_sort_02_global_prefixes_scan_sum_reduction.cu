#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Общие константы процедуры radix sort ( зеркалятся в radix_sort_03_global_prefixes_scan_accumulation.cu ):
// префикс-суммы по массиву счётчиков считаются за два прохода: в каждом блоке из RADIX_SCAN_BLOCK элементов
// вычисляется локальный эксклюзивный префикс (этот кернел) и сумма блока, потом радикальные суммы блоков
// прибавляются к локальным префиксам (следующий кернел)
#define RADIX_SCAN_BLOCK 1024u

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    const unsigned int* zeros_counts,
          unsigned int* prefixes,     // локальные (внутри блока) эксклюзивные префикс-суммы счётчиков
          unsigned int* partial_sums, // сумма счётчиков каждого блока
    unsigned int tiles_count)
{
    __shared__ unsigned int s[RADIX_SCAN_BLOCK];

    const unsigned int idx = blockIdx.x * RADIX_SCAN_BLOCK + threadIdx.x;
    unsigned int own = 0;
    if (idx < tiles_count) {
        own = zeros_counts[idx];
    }
    s[threadIdx.x] = own;
    __syncthreads();
    for (unsigned int stride = 1; stride < RADIX_SCAN_BLOCK; stride <<= 1) {
        unsigned int add = 0;
        __syncthreads();
        if (threadIdx.x >= stride) {
            add = s[threadIdx.x - stride];
        }
        __syncthreads();
        s[threadIdx.x] += add;
    }
    __syncthreads();

    if (idx < tiles_count) {
        prefixes[idx] = s[threadIdx.x] - own; // инклюзивный скан превращаем в эксклюзивный
    }
    if (threadIdx.x == RADIX_SCAN_BLOCK - 1) {
        partial_sums[blockIdx.x] = s[RADIX_SCAN_BLOCK - 1]; // вне границ занулены, так что это общая сумма блока
    }
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &zeros_counts, gpu::gpu_mem_32u &prefixes, gpu::gpu_mem_32u &partial_sums, unsigned int tiles_count)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        zeros_counts.cuptr(), prefixes.cuptr(), partial_sums.cuptr(), tiles_count);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
