// Host-RAM baseline for a pair CUDA will not pair: cudaMemcpyPeer from B into A (CUDA stages it through host RAM
// when it offers no peer access), bandwidth and one 4-byte copy, timed on B's GPU with events.
// Build: nvcc -O2 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_86,code=sm_86 staged_baseline.cu -o staged_baseline
// usage: staged_baseline <A> <B> [MiB]   (CUDA device numbers; set CUDA_DEVICE_ORDER=PCI_BUS_ID)
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                                \
  do {                                                                                       \
    cudaError_t e_ = (x);                                                                    \
    if (e_ != cudaSuccess) {                                                                 \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));   \
      exit(1);                                                                               \
    }                                                                                        \
  } while (0)

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s <A> <B> [MiB]\n", argv[0]); return 2; }
  int A = atoi(argv[1]), B = atoi(argv[2]);
  size_t bytes = (size_t)(argc > 3 ? atoll(argv[3]) : 64) << 20;
  int can = 0;
  CK(cudaDeviceCanAccessPeer(&can, B, A));
  cudaDeviceProp pa, pb;
  CK(cudaGetDeviceProperties(&pa, A));
  CK(cudaGetDeviceProperties(&pb, B));
  printf("A = %s, B = %s; CUDA peer access B->A: %s\n", pa.name, pb.name, can ? "yes" : "no");

  void *srcB, *dstA;
  CK(cudaSetDevice(B));
  CK(cudaMalloc(&srcB, bytes));
  CK(cudaMemset(srcB, 0x5a, bytes));
  CK(cudaSetDevice(A));
  CK(cudaMalloc(&dstA, bytes));
  CK(cudaMemset(dstA, 0, bytes));

  CK(cudaSetDevice(B));
  cudaStream_t s;
  cudaEvent_t e0, e1;
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaMemcpyPeerAsync(dstA, A, srcB, B, bytes, s));
  CK(cudaStreamSynchronize(s));
  const int reps = 10;
  float ms = 0;
  CK(cudaEventRecord(e0, s));
  for (int r = 0; r < reps; r++) CK(cudaMemcpyPeerAsync(dstA, A, srcB, B, bytes, s));
  CK(cudaEventRecord(e1, s));
  CK(cudaEventSynchronize(e1));
  CK(cudaEventElapsedTime(&ms, e0, e1));
  printf("cudaMemcpyPeer B -> A: %6.2f GB/s (%s)\n", reps * (double)bytes / (ms * 1e-3) / 1e9,
         can ? "CUDA peer access" : "staged through host RAM");

  std::vector<unsigned char> h(bytes);
  CK(cudaSetDevice(A));
  CK(cudaMemcpy(h.data(), dstA, bytes, cudaMemcpyDeviceToHost));
  size_t bad = 0;
  for (unsigned char c : h) bad += c != 0x5a;
  printf("checked on A: %s (%zu wrong bytes)\n", bad ? "WRONG" : "OK", bad);

  CK(cudaSetDevice(B));
  const int lat_reps = 10000;
  for (int pass = 0; pass < 2; pass++) {  // the first pass warms up
    CK(cudaEventRecord(e0, s));
    for (int r = 0; r < lat_reps; r++) CK(cudaMemcpyPeerAsync(dstA, A, srcB, B, 4, s));
    CK(cudaEventRecord(e1, s));
    CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&ms, e0, e1));
  }
  printf("4-byte cudaMemcpyPeer: %6.2f us\n", ms * 1000 / lat_reps);
  return bad ? 1 : 0;
}
