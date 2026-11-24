// P2P without CUDA peer access: GPU B reaches GPU A's memory through A's BAR1, registered on B as I/O memory.
// Works between GPUs that CUDA will not pair (for example Turing + Ampere), on a driver that lets BAR1 be registered.
// Build: nvcc -O2 -arch=native p2p_iomem.cu -o p2p_iomem -lvulkan
//
// usage: p2p_iomem <A> <B> [MiB]      (CUDA device numbers; A owns the memory, B writes and reads it)
//
// How: Vulkan allocates device-local, host-visible memory on A (VRAM that the driver maps through A's BAR1) and maps
// it into this process. CUDA on A imports the same memory as an ordinary device buffer. CUDA on B registers the
// mapping with cudaHostRegister(cudaHostRegisterIoMemory), which hands B the BAR1 bus address: B's kernels and copy
// engine then reach A's VRAM straight over PCIe. CUDA's peer-access check is never asked.
//
// NVIDIA's driver refuses that registration ("overlaps with FB or GPU BAR2" in the kernel log):
// osCheckGpuBarsOverlapAddrRange() in osmemdesc.c rejects ranges inside any GPU's BAR1, because "full-duplex P2P over
// BAR1" can deadlock. The fork this guide uses lets it through with the registry key RMPeerIoMemBar1=1 (for a trial
// load: EXTRA_NVIDIA_OPTS="NVreg_RegistryDwords=RMPeerIoMemBar1=1"); docs/mixed-generations.md has the steps.
// It also needs root: NVIDIA's driver lets only administrators register I/O memory, unless its own key
// PeerMappingOverride=1 opens that to every user (osCreateOsDescriptorFromIoMemory, same file). Run it with sudo.
// This tool only moves data one way at a time; NO_READ=1 for two copies writing against each other.
// GPU numbers follow CUDA's order: set CUDA_DEVICE_ORDER=PCI_BUS_ID to number the cards as nvidia-smi does.
//
// Checks: every word B writes is compared on A, every word B reads is compared on B. Exit status 1 on any mismatch.
// For comparison it also times cudaMemcpyPeer from B to A, which CUDA stages through host RAM for a pair without
// peer access. Run with the patched libcuda copy on LD_LIBRARY_PATH (route 1 in docs/mixed-generations.md) and CUDA
// offers peer access: the tool turns it on, so the same steps time route 1 against route 2.
// NO_READ=1 skips the read test (for writes in both directions at once from two processes).

#include <cuda_runtime.h>
#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));       \
      exit(1);                                                                                   \
    }                                                                                            \
  } while (0)
#define VK(x)                                                                                    \
  do {                                                                                           \
    VkResult r_ = (x);                                                                           \
    if (r_ != VK_SUCCESS) {                                                                      \
      fprintf(stderr, "%s:%d %s: VkResult %d\n", __FILE__, __LINE__, #x, (int)r_);              \
      exit(1);                                                                                   \
    }                                                                                            \
  } while (0)

__global__ void fill(unsigned* p, size_t n, unsigned salt) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = (unsigned)i ^ salt;
}

__global__ void verify(const unsigned* p, size_t n, unsigned salt, unsigned long long* bad) {
  unsigned long long nbad = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    nbad += p[i] != ((unsigned)i ^ salt);
  if (nbad) atomicAdd(bad, nbad);
}

static double secs(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static unsigned long long check_on(int dev, const unsigned* p, size_t n, unsigned salt) {
  CK(cudaSetDevice(dev));
  unsigned long long* bad;
  unsigned long long h = 0;
  CK(cudaMalloc(&bad, sizeof h));
  CK(cudaMemset(bad, 0, sizeof h));
  verify<<<256, 256>>>(p, n, salt, bad);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(&h, bad, sizeof h, cudaMemcpyDeviceToHost));
  CK(cudaFree(bad));
  return h;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <A> <B> [MiB]   (A owns the memory, B writes and reads it)\n", argv[0]);
    return 2;
  }
  int A = atoi(argv[1]), B = atoi(argv[2]);
  size_t bytes = (size_t)(argc > 3 ? atoi(argv[3]) : 64) << 20;
  size_t n = bytes / 4;

  cudaDeviceProp pa, pb;
  CK(cudaGetDeviceProperties(&pa, A));
  CK(cudaGetDeviceProperties(&pb, B));
  int can = 0;
  CK(cudaDeviceCanAccessPeer(&can, B, A));
  printf("A = %s (%02x:00, sm_%d%d), B = %s (%02x:00, sm_%d%d); CUDA peer access B->A: %s\n", pa.name, pa.pciBusID,
         pa.major, pa.minor, pb.name, pb.pciBusID, pb.major, pb.minor, can ? "yes" : "no");

  // Vulkan: find A by its UUID, allocate exportable device-local + host-visible memory, map it.
  VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  VkInstance inst;
  VK(vkCreateInstance(&ici, nullptr, &inst));
  uint32_t count = 0;
  VK(vkEnumeratePhysicalDevices(inst, &count, nullptr));
  VkPhysicalDevice devs[16];
  if (count > 16) count = 16;
  VK(vkEnumeratePhysicalDevices(inst, &count, devs));
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  for (uint32_t i = 0; i < count; i++) {
    VkPhysicalDeviceIDProperties idp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &idp;
    vkGetPhysicalDeviceProperties2(devs[i], &p2);
    if (!memcmp(idp.deviceUUID, pa.uuid.bytes, VK_UUID_SIZE)) phys = devs[i];
  }
  if (!phys) {
    fprintf(stderr, "no Vulkan device with the UUID of CUDA device %d\n", A);
    return 1;
  }
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(phys, &mp);
  const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  uint32_t type = UINT32_MAX;
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((mp.memoryTypes[i].propertyFlags & want) == want) {
      type = i;
      break;
    }
  if (type == UINT32_MAX) {
    fprintf(stderr, "A has no device-local + host-visible memory type (BAR1 smaller than VRAM?)\n");
    return 1;
  }
  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = 0;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  const char* exts[] = {VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME};
  VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = 1;
  dci.ppEnabledExtensionNames = exts;
  VkDevice vdev;
  VK(vkCreateDevice(phys, &dci, nullptr, &vdev));
  VkExportMemoryAllocateInfo exp = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
  exp.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
  VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.pNext = &exp;
  mai.allocationSize = bytes;
  mai.memoryTypeIndex = type;
  VkDeviceMemory mem;
  VK(vkAllocateMemory(vdev, &mai, nullptr, &mem));
  auto getFd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(vdev, "vkGetMemoryFdKHR");
  VkMemoryGetFdInfoKHR gfi = {VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
  gfi.memory = mem;
  gfi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
  int fd = -1;
  VK(getFd(vdev, &gfi, &fd));
  void* host = nullptr;
  VK(vkMapMemory(vdev, mem, 0, bytes, 0, &host));
  printf("Vulkan: %zu MiB on A, memory type %u, mapped through A's BAR1 at %p\n", bytes >> 20, type, host);

  // A: the same memory as a local CUDA buffer.
  CK(cudaSetDevice(A));
  cudaExternalMemoryHandleDesc emd = {};
  emd.type = cudaExternalMemoryHandleTypeOpaqueFd;
  emd.handle.fd = fd;
  emd.size = bytes;
  cudaExternalMemory_t ext;
  CK(cudaImportExternalMemory(&ext, &emd));
  cudaExternalMemoryBufferDesc bd = {};
  bd.size = bytes;
  void* local = nullptr;
  CK(cudaExternalMemoryGetMappedBuffer(&local, ext, &bd));

  // B: the BAR1 mapping as I/O memory.
  CK(cudaSetDevice(B));
  cudaError_t reg = cudaHostRegister(host, bytes, cudaHostRegisterIoMemory | cudaHostRegisterMapped);
  if (reg != cudaSuccess) {
    printf("B cannot register A's BAR1 as I/O memory: %s (%s)\n", cudaGetErrorString(reg),
           reg == cudaErrorNotPermitted ? "the driver allows I/O memory only to root: run it with sudo"
                                        : "the driver refuses BAR1 ranges without RMPeerIoMemBar1=1; see the header");
    return 1;
  }
  void* remote = nullptr;
  CK(cudaHostGetDevicePointer(&remote, host, 0));
  printf("B registered A's BAR1: device pointer %p\n", remote);

  int failures = 0;
  // 1. B's kernels write into A's memory; A checks.
  CK(cudaSetDevice(A));
  CK(cudaMemset(local, 0, bytes));
  CK(cudaDeviceSynchronize());
  CK(cudaSetDevice(B));
  fill<<<128, 512>>>((unsigned*)remote, n, 0x5a5a5a5au);  // first pass warms up
  CK(cudaDeviceSynchronize());
  auto t0 = std::chrono::steady_clock::now();
  const int reps = 10;
  for (int r = 0; r < reps; r++) fill<<<128, 512>>>((unsigned*)remote, n, 0x5a5a5a5au);
  CK(cudaDeviceSynchronize());
  double gbs = reps * (double)bytes / secs(t0) / 1e9;
  unsigned long long bad = check_on(A, (const unsigned*)local, n, 0x5a5a5a5au);
  printf("B kernels write A:     %6.2f GB/s, checked on A: %s (%llu wrong words)\n", gbs, bad ? "WRONG" : "OK", bad);
  failures += bad != 0;

  // 2. B's copy engine writes into A's memory (B's own buffer -> the I/O mapping); A checks.
  void* srcB;
  CK(cudaSetDevice(B));
  CK(cudaMalloc(&srcB, bytes));
  fill<<<128, 512>>>((unsigned*)srcB, n, 0xc3c3c3c3u);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(remote, srcB, bytes, cudaMemcpyDeviceToDevice));
  t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; r++) CK(cudaMemcpyAsync(remote, srcB, bytes, cudaMemcpyDeviceToDevice));
  CK(cudaDeviceSynchronize());
  gbs = reps * (double)bytes / secs(t0) / 1e9;
  bad = check_on(A, (const unsigned*)local, n, 0xc3c3c3c3u);
  printf("B copy engine -> A:    %6.2f GB/s, checked on A: %s (%llu wrong words)\n", gbs, bad ? "WRONG" : "OK", bad);
  failures += bad != 0;

  // 3. A writes a pattern locally; B's kernels read it over PCIe and check (one direction only).
  if (!getenv("NO_READ")) {
    CK(cudaSetDevice(A));
    fill<<<128, 512>>>((unsigned*)local, n, 0x3c3c3c3cu);
    CK(cudaDeviceSynchronize());
    CK(cudaSetDevice(B));
    t0 = std::chrono::steady_clock::now();
    bad = check_on(B, (const unsigned*)remote, n, 0x3c3c3c3cu);
    gbs = (double)bytes / secs(t0) / 1e9;
    printf("B kernels read A:      %6.2f GB/s, checked on B: %s (%llu wrong words)\n", gbs, bad ? "WRONG" : "OK", bad);
    failures += bad != 0;
  }

  // 4. For comparison: cudaMemcpyPeer B -> A, which CUDA stages through host RAM when it offers no peer access. With
  //    peer access (route 1: the patched libcuda copy on LD_LIBRARY_PATH) it is turned on, so steps 4 and 5 time it.
  if (can) {
    CK(cudaSetDevice(B));
    CK(cudaDeviceEnablePeerAccess(A, 0));
    CK(cudaSetDevice(A));
    CK(cudaDeviceEnablePeerAccess(B, 0));
  }
  void* dstA;
  CK(cudaSetDevice(A));
  CK(cudaMalloc(&dstA, bytes));
  CK(cudaMemcpyPeer(dstA, A, srcB, B, bytes));
  CK(cudaDeviceSynchronize());
  t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; r++) CK(cudaMemcpyPeer(dstA, A, srcB, B, bytes));
  CK(cudaSetDevice(A));
  CK(cudaDeviceSynchronize());
  gbs = reps * (double)bytes / secs(t0) / 1e9;
  bad = check_on(A, (const unsigned*)dstA, n, 0xc3c3c3c3u);
  printf("cudaMemcpyPeer B -> A: %6.2f GB/s (%s), checked on A: %s\n", gbs,
         can ? "CUDA peer access" : "staged through host RAM", bad ? "WRONG" : "OK");
  failures += bad != 0;
  CK(cudaFree(dstA));

  // 5. Latency of one small copy, timed on B's GPU with events as p2pBandwidthLatencyTest does: 4 bytes into A's
  //    memory through the BAR1 mapping, and the same through cudaMemcpyPeer.
  CK(cudaSetDevice(A));
  CK(cudaMalloc(&dstA, 4));
  CK(cudaSetDevice(B));
  cudaStream_t s;
  cudaEvent_t e0, e1;
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  const int lat_reps = 10000;
  float ms_bar1 = 0, ms_peer = 0;
  for (int pass = 0; pass < 2; pass++) {  // the first pass warms up
    CK(cudaEventRecord(e0, s));
    for (int r = 0; r < lat_reps; r++) CK(cudaMemcpyAsync(remote, srcB, 4, cudaMemcpyDeviceToDevice, s));
    CK(cudaEventRecord(e1, s));
    CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&ms_bar1, e0, e1));
    CK(cudaEventRecord(e0, s));
    for (int r = 0; r < lat_reps; r++) CK(cudaMemcpyPeerAsync(dstA, A, srcB, B, 4, s));
    CK(cudaEventRecord(e1, s));
    CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&ms_peer, e0, e1));
  }
  printf("4-byte copy latency:   %6.2f us through A's BAR1, %6.2f us with cudaMemcpyPeer (%s)\n",
         ms_bar1 * 1000 / lat_reps, ms_peer * 1000 / lat_reps, can ? "CUDA peer access" : "staged through host RAM");
  CK(cudaSetDevice(A));
  CK(cudaFree(dstA));

  CK(cudaHostUnregister(host));
  vkUnmapMemory(vdev, mem);
  printf("%s\n", failures ? "FAIL" : "ALL OK");
  return failures ? 1 : 0;
}
