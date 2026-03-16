#include <cstdio>
#include <cuda_runtime.h>
#include "CudaAllocator.h"
#include "helper_cuda.h"
#include <cmath>
#include <vector>

// 改为网格跨步循环（Grid-stride loop）
// 数组类型改为 float，因为 sinf 返回的是浮点数
__global__ void fill_sin(float *arr, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        arr[i] = sinf(i);
    }
}

// 使用 atomicAdd 解决竞态条件
__global__ void filter_positive(int *counter, float *res, float const *arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;           // 修复：>= 而不是 
    if (arr[i] >= 0) {
        int loc = atomicAdd(counter, 1);  // 原子操作，保证线程安全
        res[loc] = arr[i];                // 修复：存 arr[i] 而不是 n
    }
}

int main() {
    constexpr int n = 1 << 24;
    std::vector<float, CudaAllocator<float>> arr(n);
    std::vector<float, CudaAllocator<float>> res(n);
    std::vector<int, CudaAllocator<int>> counter(1, 0);  // 初始化为 0

    // 网格跨步循环后，参数不再必须是 n/1024
    // 32 个 block，每块 128 线程，每线程循环处理多个元素
    fill_sin<<<32, 128>>>(arr.data(), n);

    // 修复：用 ceil 除法保证覆盖所有元素（边角料问题）
    filter_positive<<<(n + 1023) / 1024, 1024>>>(counter.data(), res.data(), arr.data(), n);

    // 修复：CPU 访问前必须同步
    checkCudaErrors(cudaDeviceSynchronize());

    if (counter[0] <= n / 50) {
        printf("Result too short! %d <= %d\n", counter[0], n / 50);
        return -1;
    }
    for (int i = 0; i < counter[0]; i++) {
        if (res[i] < 0) {
            printf("Wrong At %d: %f < 0\n", i, res[i]);
            return -1;
        }
    }

    printf("All Correct!\n");
    return 0;
}