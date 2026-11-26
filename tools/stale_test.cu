// Stale-step test for NCCL between two GPUs of one process: thousands of all-reduces back to back with no host
// synchronisation, each with its own data, every element of every result checked on the GPU.
// Iteration i on rank r fills sendbuff with a pattern of (i, r, index) and recvbuff with a sentinel (out-of-place), the
// all-reduce runs, and a kernel compares each element with the exact sum (small integers: exact in float and bf16),
// counting mismatches. A step read before its data has landed shows up as an older iteration's values or the sentinel.
// nccl-tests' check mode starts each checked collective from idle; this keeps the pipeline full, like a real program.
//
// usage: stale_test <min bytes> <max bytes> <iterations per size> [f32|bf16] [inplace]
// prints one line per size: "size <bytes>: <wrong> wrong of <elements x iterations> elements, <us> us per all-reduce"
// exit 0 when nothing was wrong, 1 when something was, 2 on an error.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(2); } } while (0)
#define NK(x) do { ncclResult_t e = (x); if (e != ncclSuccess) { fprintf(stderr, "NCCL %s at %s:%d\n", ncclGetErrorString(e), __FILE__, __LINE__); exit(2); } } while (0)

__device__ __forceinline__ int pattern(unsigned it, int rank, size_t i, int mod) {
  return (int)((it * 131u + (unsigned)(i * 7u) + (unsigned)rank * 1009u) % (unsigned)mod);
}
__device__ __forceinline__ float toF(float x) { return x; }
__device__ __forceinline__ float toF(__nv_bfloat16 x) { return __bfloat162float(x); }
template <typename T> __device__ __forceinline__ T fromF(float x);
template <> __device__ __forceinline__ float fromF<float>(float x) { return x; }
template <> __device__ __forceinline__ __nv_bfloat16 fromF<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

template <typename T>
__global__ void fill(T* send, T* recv, size_t n, unsigned it, int rank, int mod) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    send[i] = fromF<T>((float)pattern(it, rank, i, mod));
    if (recv) recv[i] = fromF<T>(-1.0f);
  }
}

template <typename T>
__global__ void check(const T* recv, size_t n, unsigned it, int mod, unsigned long long* wrong) {
  unsigned long long w = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float want = (float)(pattern(it, 0, i, mod) + pattern(it, 1, i, mod));
    if (toF(recv[i]) != want) w++;
  }
  if (w) atomicAdd(wrong, w);
}

template <typename T>
int run(size_t minB, size_t maxB, int iters, bool inplace, ncclDataType_t dt, int mod) {
  int dev[2] = {0, 1};
  ncclComm_t comm[2];
  NK(ncclCommInitAll(comm, 2, dev));
  cudaStream_t st[2];
  T *send[2], *recv[2];
  unsigned long long* wrong[2];
  int blocks[2];
  for (int r = 0; r < 2; r++) {
    CK(cudaSetDevice(r));
    CK(cudaStreamCreateWithFlags(&st[r], cudaStreamNonBlocking));
    CK(cudaMalloc(&send[r], maxB));
    CK(cudaMalloc(&recv[r], maxB));
    CK(cudaMalloc(&wrong[r], sizeof(unsigned long long)));
    int sms = 80;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, r));
    blocks[r] = 2 * sms;
  }
  int bad = 0;
  unsigned it = 1;
  for (size_t bytes = minB; bytes <= maxB; bytes *= 2) {
    size_t n = bytes / sizeof(T);
    for (int r = 0; r < 2; r++) {
      CK(cudaSetDevice(r));
      CK(cudaMemsetAsync(wrong[r], 0, sizeof(unsigned long long), st[r]));
    }
    for (int r = 0; r < 2; r++) { CK(cudaSetDevice(r)); CK(cudaStreamSynchronize(st[r])); }
    auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < iters; k++, it++) {
      for (int r = 0; r < 2; r++) {
        CK(cudaSetDevice(r));
        fill<T><<<blocks[r], 256, 0, st[r]>>>(send[r], inplace ? nullptr : recv[r], n, it, r, mod);
      }
      NK(ncclGroupStart());
      for (int r = 0; r < 2; r++) NK(ncclAllReduce(send[r], inplace ? send[r] : recv[r], n, dt, ncclSum, comm[r], st[r]));
      NK(ncclGroupEnd());
      for (int r = 0; r < 2; r++) {
        CK(cudaSetDevice(r));
        check<T><<<blocks[r], 256, 0, st[r]>>>(inplace ? send[r] : recv[r], n, it, mod, wrong[r]);
      }
    }
    unsigned long long w[2] = {0, 0};
    for (int r = 0; r < 2; r++) {
      CK(cudaSetDevice(r));
      CK(cudaStreamSynchronize(st[r]));
      CK(cudaMemcpy(&w[r], wrong[r], sizeof w[r], cudaMemcpyDeviceToHost));
    }
    double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
    printf("size %zu: %llu wrong of %llu elements (rank 0 %llu, rank 1 %llu), %.1f us per all-reduce with fill and check\n",
           bytes, w[0] + w[1], 2ull * n * iters, w[0], w[1], us);
    fflush(stdout);
    if (w[0] + w[1]) bad = 1;
  }
  for (int r = 0; r < 2; r++) NK(ncclCommDestroy(comm[r]));
  return bad;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <min bytes> <max bytes> <iterations per size> [f32|bf16] [inplace]\n", argv[0]);
    return 2;
  }
  size_t minB = strtoull(argv[1], nullptr, 0), maxB = strtoull(argv[2], nullptr, 0);
  int iters = atoi(argv[3]);
  bool bf16 = argc > 4 && strcmp(argv[4], "bf16") == 0;
  bool inplace = argc > 5 && strcmp(argv[5], "inplace") == 0;
  printf("stale_test: %zu-%zu bytes, %d all-reduces per size back to back, %s, %s\n", minB, maxB, iters,
         bf16 ? "bf16 (values < 64: exact)" : "f32 (values < 2048: exact)", inplace ? "in place" : "out of place");
  int bad = bf16 ? run<__nv_bfloat16>(minB, maxB, iters, inplace, ncclBfloat16, 64)
                 : run<float>(minB, maxB, iters, inplace, ncclFloat32, 2048);
  printf("stale_test: %s\n", bad ? "WRONG RESULTS" : "every element right");
  return bad;
}
