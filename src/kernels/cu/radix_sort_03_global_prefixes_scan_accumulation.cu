#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Общие константы процедуры radix sort ( зеркалятся в radix_sort_02_global_prefixes_scan_sum_reduction.cu ):
// на входе локальные эксклюзивные префиксы и суммы блоков, на выходе - глобальный эксклюзивный префикс
#define RADIX_SCAN_BLOCK 1024u

__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    const unsigned int* partial_sums,
          unsigned int* zeros_starts, // [idx] - глобальный старт нулей тайла idx; [tiles_count] - полное число нулей (старт зоны единиц)
    unsigned int tiles_count,
    unsigned int scans_count) // количество блоков, на которые разбит массив счётчиков (оно же сколько значений в partial_sums)
{
    // Последний блок дополнительно сохраняет полную сумму всех счётчиков -
    // это старт глобальной зоны единиц (конец зоны нулей), нужен radix_sort_04_scatter
    if (blockIdx.x == scans_count - 1 && threadIdx.x == 0) {
        unsigned int total_zeros = 0;
        for (unsigned int b = 0; b < scans_count; ++b) {
            total_zeros += partial_sums[b];
        }
        zeros_starts[tiles_count] = total_zeros;
    }

    const unsigned int idx = blockIdx.x * RADIX_SCAN_BLOCK + threadIdx.x;
    if (idx >= tiles_count) {
        return;
    }

    unsigned int offset = 0;
    for (unsigned int b = 0; b < blockIdx.x; ++b) {
        offset += partial_sums[b];
    }
    zeros_starts[idx] += offset;
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &partial_sums, gpu::gpu_mem_32u &zeros_starts, unsigned int tiles_count, unsigned int scans_count)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        partial_sums.cuptr(), zeros_starts.cuptr(), tiles_count, scans_count);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
