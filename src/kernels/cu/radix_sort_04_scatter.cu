#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

// Общие константы процедуры radix sort ( зеркалятся в radix_sort_01_local_counting.cu );
// раскладка элементов по нитям обязана совпадать с radix_sort_01_local_counting
#define RADIX_ITEMS_PER_THREAD 16u
#define RADIX_TILE (GROUP_SIZE * RADIX_ITEMS_PER_THREAD)

__global__ void radix_sort_04_scatter(
    const unsigned int* input,
    const unsigned int* zeros_starts, // глобальные старты нулей тайлов; zeros_starts[tiles_count] - полное число нулей, т.е. старт глобальной зоны единиц
    unsigned int* output,
    unsigned int n,
    unsigned int tiles_count,
    unsigned int digit) // номер бита (0..31)
{
    __shared__ unsigned int s[GROUP_SIZE];

    const unsigned int base = blockIdx.x * RADIX_TILE + threadIdx.x * RADIX_ITEMS_PER_THREAD;

    // Фаза 1: локальные ранги. Каждая нить считает нули в своём куске,
    // потом блок-скан даёт число нулей до начала моего куска и во всём тайле
    unsigned int zeros_mine = 0;
    for (unsigned int k = 0; k < RADIX_ITEMS_PER_THREAD; ++k) {
        const unsigned int idx = base + k;
        if (idx < n) {
            const unsigned int bit = (input[idx] >> digit) & 1u;
            zeros_mine += (bit ^ 1u);
        }
    }
    s[threadIdx.x] = zeros_mine;
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

    const unsigned int zeros_before = s[threadIdx.x] - zeros_mine; // нулей до начала моего куска в тайле
    const unsigned int ones_before = threadIdx.x * RADIX_ITEMS_PER_THREAD - zeros_before;

    const unsigned int zeros_start = zeros_starts[blockIdx.x] + zeros_before;
    // Старт зоны единиц глобально: конец зоны нулей + сколько единиц в предыдущих тайлах (разность
    // глобального начала тайла и его глобального старта нулей) + сколько единиц до моего куска в тайле
    const unsigned int ones_before_tile = blockIdx.x * RADIX_TILE - zeros_starts[blockIdx.x];
    const unsigned int ones_start = zeros_starts[tiles_count] + ones_before_tile + ones_before;
    unsigned int r0 = zeros_start;
    unsigned int r1 = ones_start;

    // Фаза 2: перестановка элементов: сначала элементы с нулём в разряде, потом с единицей;
    // относительный порядок среди равных сохраняется (устойчивость)
    for (unsigned int k = 0; k < RADIX_ITEMS_PER_THREAD; ++k) {
        const unsigned int idx = base + k;
        if (idx >= n) {
            break;
        }
        const unsigned int value = input[idx];
        const unsigned int bit = (value >> digit) & 1u;
        if (bit) {
            output[r1] = value;
            ++r1;
        } else {
            output[r0] = value;
            ++r0;
        }
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input, const gpu::gpu_mem_32u& zeros_starts, gpu::gpu_mem_32u& output, unsigned int n, unsigned int tiles_count, unsigned int digit)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        input.cuptr(), zeros_starts.cuptr(), output.cuptr(), n, tiles_count, digit);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
