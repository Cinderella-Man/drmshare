// Memory sharing between two GPUs through a dma-buf (not CUDA P2P). Vulkan on GPU A allocates device-local memory and
// exports it as a dma-buf (VK_EXT_external_memory_dma_buf); Vulkan on GPU B imports it, and B's copy engine writes into
// A's memory and reads it back. Every word is checked on the other GPU. No CUDA peer access and no I/O-memory
// registration involved.
// Between two NVIDIA GPUs the import needs nvidia-drm with peer_vidmem_dmabuf=1 (docs/volta-dmabuf.md); on a stock
// nvidia-drm it stops at "B cannot import A's dma-buf: no common memory type".
//
// Build: g++ -O2 -std=c++17 p2p_dmabuf.cpp -o p2p_dmabuf -lvulkan
// usage: p2p_dmabuf <A> <B> [MiB]   (NVIDIA GPUs numbered in PCI bus order, as nvidia-smi does; A owns the memory)
// NO_READ=1 times only B's writes into A (for two copies writing against each other: both ways at once).
// BARRIER=<dir>: before the timed writes, create <dir>/ready.<A><B> and wait until <dir>/go exists.
#include <unistd.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define VK(x)                                                                         \
  do {                                                                                \
    VkResult r_ = (x);                                                                \
    if (r_ != VK_SUCCESS) {                                                           \
      fprintf(stderr, "%s:%d %s failed: VkResult %d\n", __FILE__, __LINE__, #x, (int)r_); \
      exit(2);                                                                        \
    }                                                                                 \
  } while (0)

struct Gpu {
  VkPhysicalDevice pd{};
  VkDevice dev{};
  VkQueue q{};
  uint32_t qf = 0;
  VkCommandPool pool{};
  float tsPeriod = 1;
  std::string name;
  VkPhysicalDevicePCIBusInfoPropertiesEXT pci{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
  VkPhysicalDeviceMemoryProperties mp{};
  PFN_vkGetMemoryFdKHR getFd{};
  PFN_vkGetMemoryFdPropertiesKHR getFdProps{};
};

static const char* kExt[] = {"VK_KHR_external_memory_fd", "VK_EXT_external_memory_dma_buf"};

static void open_gpu(Gpu& g) {
  VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  p2.pNext = &g.pci;
  vkGetPhysicalDeviceProperties2(g.pd, &p2);
  g.name = p2.properties.deviceName;
  g.tsPeriod = p2.properties.limits.timestampPeriod;
  vkGetPhysicalDeviceMemoryProperties(g.pd, &g.mp);
  uint32_t n = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(g.pd, &n, nullptr);
  std::vector<VkQueueFamilyProperties> qfs(n);
  vkGetPhysicalDeviceQueueFamilyProperties(g.pd, &n, qfs.data());
  g.qf = UINT32_MAX;
  for (uint32_t i = 0; i < n; i++)
    if ((qfs[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) && qfs[i].timestampValidBits) {
      g.qf = i;
      break;
    }
  if (g.qf == UINT32_MAX) { fprintf(stderr, "no usable queue family on %s\n", g.name.c_str()); exit(2); }
  float prio = 1;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = g.qf;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = 2;
  dci.ppEnabledExtensionNames = kExt;
  VK(vkCreateDevice(g.pd, &dci, nullptr, &g.dev));
  vkGetDeviceQueue(g.dev, g.qf, 0, &g.q);
  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.queueFamilyIndex = g.qf;
  VK(vkCreateCommandPool(g.dev, &pci, nullptr, &g.pool));
  g.getFd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(g.dev, "vkGetMemoryFdKHR");
  g.getFdProps = (PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(g.dev, "vkGetMemoryFdPropertiesKHR");
}

static std::string flags_str(VkMemoryPropertyFlags f) {
  std::string s;
  if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) s += "DEVICE_LOCAL ";
  if (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) s += "HOST_VISIBLE ";
  if (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) s += "HOST_COHERENT ";
  if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) s += "HOST_CACHED ";
  return s.empty() ? "none" : s;
}

// Lowest-numbered type in `bits` that has all of `want` and none of `avoid`.
static uint32_t pick_type(const Gpu& g, uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
  for (uint32_t i = 0; i < g.mp.memoryTypeCount; i++) {
    auto f = g.mp.memoryTypes[i].propertyFlags;
    if ((bits & (1u << i)) && (f & want) == want && !(f & avoid)) return i;
  }
  return UINT32_MAX;
}

struct Buf {
  VkBuffer b{};
  VkDeviceMemory m{};
  uint32_t type = 0;
};

static VkBuffer make_buffer(Gpu& g, VkDeviceSize size, bool external) {
  VkExternalMemoryBufferCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
  ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.pNext = external ? &ext : nullptr;
  bci.size = size;
  bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VkBuffer b;
  VK(vkCreateBuffer(g.dev, &bci, nullptr, &b));
  return b;
}

// Plain buffer (exportable as a dma-buf when `exportable`) in a type with `want` and without `avoid`.
static Buf alloc_buffer(Gpu& g, VkDeviceSize size, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid,
                        bool exportable) {
  Buf r;
  r.b = make_buffer(g, size, exportable);
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(g.dev, r.b, &req);
  r.type = pick_type(g, req.memoryTypeBits, want, avoid);
  if (r.type == UINT32_MAX) r.type = pick_type(g, req.memoryTypeBits, want, 0);
  if (r.type == UINT32_MAX) { fprintf(stderr, "no memory type with %s on %s\n", flags_str(want).c_str(), g.name.c_str()); exit(2); }
  VkExportMemoryAllocateInfo exp{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
  exp.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.pNext = exportable ? &exp : nullptr;
  mai.allocationSize = req.size;
  mai.memoryTypeIndex = r.type;
  VK(vkAllocateMemory(g.dev, &mai, nullptr, &r.m));
  VK(vkBindBufferMemory(g.dev, r.b, r.m, 0));
  return r;
}

static VkCommandBuffer begin(Gpu& g) {
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = g.pool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VkCommandBuffer cb;
  VK(vkAllocateCommandBuffers(g.dev, &ai, &cb));
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK(vkBeginCommandBuffer(cb, &bi));
  return cb;
}

static void finish(Gpu& g, VkCommandBuffer cb) {
  VK(vkEndCommandBuffer(cb));
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence f;
  VK(vkCreateFence(g.dev, &fci, nullptr, &f));
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb;
  VK(vkQueueSubmit(g.q, 1, &si, f));
  VK(vkWaitForFences(g.dev, 1, &f, VK_TRUE, UINT64_MAX));
  vkDestroyFence(g.dev, f, nullptr);
  vkFreeCommandBuffers(g.dev, g.pool, 1, &cb);
}

static void barrier(VkCommandBuffer cb) {
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0,
                       nullptr);
}

// Timed: `reps` repetitions of `op`, separated by barriers. Returns seconds per repetition.
template <class F>
static double timed(Gpu& g, int reps, F op) {
  VkQueryPoolCreateInfo qp{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  qp.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qp.queryCount = 2;
  VkQueryPool pool;
  VK(vkCreateQueryPool(g.dev, &qp, nullptr, &pool));
  VkCommandBuffer cb = begin(g);
  vkCmdResetQueryPool(cb, pool, 0, 2);
  op(cb);  // warm-up
  barrier(cb);
  vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, 0);
  for (int r = 0; r < reps; r++) {
    op(cb);
    barrier(cb);
  }
  vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 1);
  finish(g, cb);
  uint64_t ts[2];
  VK(vkGetQueryPoolResults(g.dev, pool, 0, 2, sizeof ts, ts, sizeof(uint64_t),
                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
  vkDestroyQueryPool(g.dev, pool, nullptr);
  return (ts[1] - ts[0]) * (double)g.tsPeriod * 1e-9 / reps;
}

static uint32_t pattern(uint64_t i, uint32_t seed) { return (uint32_t)(i * 2654435761u) ^ seed; }

// Host-visible staging buffer on `g`, `n` words.
static Buf staging(Gpu& g, VkDeviceSize bytes) {
  return alloc_buffer(g, bytes, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0, false);
}

static void upload(Gpu& g, Buf& dst, VkDeviceSize bytes, uint32_t seed) {
  Buf st = staging(g, bytes);
  uint32_t* p;
  VK(vkMapMemory(g.dev, st.m, 0, bytes, 0, (void**)&p));
  for (uint64_t i = 0; i < bytes / 4; i++) p[i] = pattern(i, seed);
  vkUnmapMemory(g.dev, st.m);
  VkCommandBuffer cb = begin(g);
  VkBufferCopy c{0, 0, bytes};
  vkCmdCopyBuffer(cb, st.b, dst.b, 1, &c);
  finish(g, cb);
  vkDestroyBuffer(g.dev, st.b, nullptr);
  vkFreeMemory(g.dev, st.m, nullptr);
}

// Copies `src` to a staging buffer on `g` and counts words that differ from pattern(seed).
static uint64_t check(Gpu& g, Buf& src, VkDeviceSize bytes, uint32_t seed) {
  Buf st = staging(g, bytes);
  VkCommandBuffer cb = begin(g);
  VkBufferCopy c{0, 0, bytes};
  vkCmdCopyBuffer(cb, src.b, st.b, 1, &c);
  finish(g, cb);
  uint32_t* p;
  VK(vkMapMemory(g.dev, st.m, 0, bytes, 0, (void**)&p));
  uint64_t bad = 0;
  for (uint64_t i = 0; i < bytes / 4; i++) bad += p[i] != pattern(i, seed);
  vkUnmapMemory(g.dev, st.m);
  vkDestroyBuffer(g.dev, st.b, nullptr);
  vkFreeMemory(g.dev, st.m, nullptr);
  return bad;
}

static void show_dmabuf_exporter(VkDeviceSize bytes) {
  std::ifstream f("/sys/kernel/debug/dma_buf/bufinfo");
  if (!f) { printf("dma-buf exporter: /sys/kernel/debug/dma_buf/bufinfo not readable\n"); return; }
  std::string line, want = std::to_string(bytes);
  while (std::getline(f, line))
    if (line.find(want) != std::string::npos) { printf("dma-buf in the kernel: %s\n", line.c_str()); return; }
  printf("dma-buf in the kernel: no %s-byte buffer listed\n", want.c_str());
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s <A> <B> [MiB]\n", argv[0]); return 2; }
  int ia = atoi(argv[1]), ib = atoi(argv[2]);
  VkDeviceSize bytes = (argc > 3 ? atoll(argv[3]) : 64) << 20;
  const int reps = getenv("REPS") ? atoi(getenv("REPS")) : 10;

  VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  ai.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &ai;
  VkInstance inst;
  VK(vkCreateInstance(&ici, nullptr, &inst));
  uint32_t n = 0;
  vkEnumeratePhysicalDevices(inst, &n, nullptr);
  std::vector<VkPhysicalDevice> all(n);
  vkEnumeratePhysicalDevices(inst, &n, all.data());
  std::vector<Gpu> gpus;
  for (auto pd : all) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(pd, &p);
    if (p.vendorID != 0x10de) continue;
    Gpu g;
    g.pd = pd;
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &g.pci;
    vkGetPhysicalDeviceProperties2(pd, &p2);
    gpus.push_back(g);
  }
  std::sort(gpus.begin(), gpus.end(), [](const Gpu& x, const Gpu& y) {
    return std::make_tuple(x.pci.pciDomain, x.pci.pciBus, x.pci.pciDevice) <
           std::make_tuple(y.pci.pciDomain, y.pci.pciBus, y.pci.pciDevice);
  });
  if (ia == ib || ia < 0 || ib < 0 || ia >= (int)gpus.size() || ib >= (int)gpus.size()) {
    fprintf(stderr, "need two different NVIDIA GPUs (found %zu)\n", gpus.size());
    return 2;
  }
  Gpu& A = gpus[ia];
  Gpu& B = gpus[ib];
  open_gpu(A);
  open_gpu(B);
  printf("A = %s (%02x:%02x), B = %s (%02x:%02x)\n", A.name.c_str(), A.pci.pciBus, A.pci.pciDevice, B.name.c_str(),
         B.pci.pciBus, B.pci.pciDevice);

  // A: device-local memory, exported as a dma-buf. Prefer a type that is not host-visible (plain VRAM).
  Buf a = alloc_buffer(A, bytes, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, true);
  printf("A: %llu MiB in memory type %u (%s), heap %u\n", (unsigned long long)(bytes >> 20), a.type,
         flags_str(A.mp.memoryTypes[a.type].propertyFlags).c_str(), A.mp.memoryTypes[a.type].heapIndex);
  VkMemoryGetFdInfoKHR gfi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
  gfi.memory = a.m;
  gfi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
  int fd = -1;
  VK(A.getFd(A.dev, &gfi, &fd));
  printf("A: exported as dma-buf fd %d\n", fd);
  show_dmabuf_exporter(bytes);

  // How fast A fills its own exported buffer: hundreds of GB/s means VRAM, about 10 GB/s means system RAM.
  double s = timed(A, reps, [&](VkCommandBuffer cb) { vkCmdFillBuffer(cb, a.b, 0, bytes, 0); });
  printf("A fills its exported buffer:  %8.1f GB/s\n", bytes / s / 1e9);

  // B: import the dma-buf.
  VkMemoryFdPropertiesKHR fdp{VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
  VK(B.getFdProps(B.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fdp));
  Buf bi;
  bi.b = make_buffer(B, bytes, true);
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(B.dev, bi.b, &req);
  uint32_t bits = req.memoryTypeBits & fdp.memoryTypeBits;
  printf("B: import allowed in memory types 0x%x (buffer 0x%x, dma-buf 0x%x)\n", bits, req.memoryTypeBits,
         fdp.memoryTypeBits);
  if (!bits) { printf("B cannot import A's dma-buf: no common memory type\nFAIL\n"); return 1; }
  bi.type = __builtin_ctz(bits);
  printf("B: importing into memory type %u (%s)\n", bi.type, flags_str(B.mp.memoryTypes[bi.type].propertyFlags).c_str());
  VkImportMemoryFdInfoKHR imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
  imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
  imp.fd = fd;  // owned by the driver once the import succeeds
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.pNext = &imp;
  mai.allocationSize = std::max<VkDeviceSize>(req.size, bytes);
  mai.memoryTypeIndex = bi.type;
  VkResult ir = vkAllocateMemory(B.dev, &mai, nullptr, &bi.m);
  if (ir != VK_SUCCESS) { printf("B cannot import A's dma-buf: VkResult %d\nFAIL\n", (int)ir); return 1; }
  VK(vkBindBufferMemory(B.dev, bi.b, bi.m, 0));
  printf("B: imported A's memory\n");

  int failures = 0;
  // 1. B's copy engine writes into A's memory (B's own buffer -> the import); A checks.
  Buf bl = alloc_buffer(B, bytes, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, false);
  upload(B, bl, bytes, 0x1234abcd);
  VkBufferCopy whole{0, 0, bytes};
  if (const char* dir = getenv("BARRIER")) {
    std::string ready = std::string(dir) + "/ready." + argv[1] + argv[2], go = std::string(dir) + "/go";
    FILE* f = fopen(ready.c_str(), "w");
    if (f) fclose(f);
    while (access(go.c_str(), F_OK) != 0) usleep(1000);
  }
  s = timed(B, reps, [&](VkCommandBuffer cb) { vkCmdCopyBuffer(cb, bl.b, bi.b, 1, &whole); });
  uint64_t bad = check(A, a, bytes, 0x1234abcd);
  printf("B copies into A:   %6.2f GB/s, checked on A: %s (%llu wrong words)\n", bytes / s / 1e9, bad ? "WRONG" : "OK",
         (unsigned long long)bad);
  failures += bad != 0;

  if (getenv("NO_READ")) {
    printf("%s\n", failures ? "FAIL" : "ALL OK");
    return failures ? 1 : 0;
  }
  // 2. A writes a pattern locally; B's copy engine reads it (the import -> B's own buffer); B checks.
  upload(A, a, bytes, 0x5ee0cafe);
  s = timed(B, reps, [&](VkCommandBuffer cb) { vkCmdCopyBuffer(cb, bi.b, bl.b, 1, &whole); });
  bad = check(B, bl, bytes, 0x5ee0cafe);
  printf("B reads from A:    %6.2f GB/s, checked on B: %s (%llu wrong words)\n", bytes / s / 1e9, bad ? "WRONG" : "OK",
         (unsigned long long)bad);
  failures += bad != 0;

  // 3. One small copy: 4 bytes from B into A's memory, 10000 in a row.
  VkBufferCopy four{0, 0, 4};
  s = timed(B, 10000, [&](VkCommandBuffer cb) { vkCmdCopyBuffer(cb, bl.b, bi.b, 1, &four); });
  printf("4-byte copy B -> A: %6.2f us\n", s * 1e6);

  printf("%s\n", failures ? "FAIL" : "ALL OK");
  return failures ? 1 : 0;
}
