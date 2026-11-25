// BAR1 mapping probe. Vulkan allocates DEVICE_LOCAL|HOST_VISIBLE memory on one GPU (what llama.cpp's Vulkan backend
// does for every buffer when Resizable BAR is on), maps it, writes and reads one word per 4 KiB page, and keeps it
// mapped. After every block it prints the total mapped and how many NVRM lines the kernel log gained, so the point
// where the driver stops mapping shows up exactly. With static BAR1 on, a card whose window is not much larger than its
// memory stops at window minus memory (an RTX 3060 12 GB with a 16 GB window: 4 GiB); see docs/troubleshooting.md,
// "Vulkan: the kernel log fills with dmaAllocMapping_GM107".
//
// Build: g++ -O2 -std=c++17 bar1_probe.cpp -o bar1_probe -lvulkan
// usage (root, for dmesg): bar1_probe <gpu, PCI bus order like nvidia-smi> <block MiB> <blocks> [extra bytes per block]
// Example: sudo ./bar1_probe 1 1024 11      exit 0: all mapped and read back; 3: a page read back wrong
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define VK(x)                                                                              \
  do {                                                                                     \
    VkResult r_ = (x);                                                                     \
    if (r_ != VK_SUCCESS) {                                                                \
      fprintf(stderr, "%s:%d %s failed: VkResult %d\n", __FILE__, __LINE__, #x, (int)r_);  \
      return 2;                                                                            \
    }                                                                                      \
  } while (0)

static long nvrm_lines() {
  FILE* p = popen("dmesg | grep -c NVRM", "r");
  long n = -1;
  if (p) { if (fscanf(p, "%ld", &n) != 1) n = -1; pclose(p); }
  return n;
}

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s <gpu> <block MiB> <blocks> [extra bytes]\n", argv[0]); return 1; }
  int want = atoi(argv[1]);
  uint64_t block = strtoull(argv[2], nullptr, 0) << 20;
  int blocks = atoi(argv[3]);
  uint64_t extra = argc > 4 ? strtoull(argv[4], nullptr, 0) : 0;
  block += extra;

  VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  ai.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &ai;
  VkInstance inst;
  VK(vkCreateInstance(&ici, nullptr, &inst));
  uint32_t n = 0;
  vkEnumeratePhysicalDevices(inst, &n, nullptr);
  std::vector<VkPhysicalDevice> pds(n);
  vkEnumeratePhysicalDevices(inst, &n, pds.data());

  // NVIDIA devices in PCI bus order, like nvidia-smi
  struct D { VkPhysicalDevice pd; uint32_t bus; std::string name; };
  std::vector<D> nv;
  for (auto pd : pds) {
    VkPhysicalDevicePCIBusInfoPropertiesEXT pci{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &pci;
    vkGetPhysicalDeviceProperties2(pd, &p2);
    if (p2.properties.vendorID == 0x10de) nv.push_back({pd, pci.pciBus, p2.properties.deviceName});
  }
  std::sort(nv.begin(), nv.end(), [](const D& a, const D& b) { return a.bus < b.bus; });
  if (want < 0 || want >= (int)nv.size()) { fprintf(stderr, "no NVIDIA GPU %d\n", want); return 1; }
  VkPhysicalDevice pd = nv[want].pd;

  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(pd, &mp);
  const VkMemoryPropertyFlags need = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  uint32_t type = UINT32_MAX;
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((mp.memoryTypes[i].propertyFlags & need) == need) { type = i; break; }
  if (type == UINT32_MAX) { fprintf(stderr, "no host-visible device-local memory type (resizable BAR off?)\n"); return 1; }
  printf("GPU %d %s (bus %02x): memory type %u heap %u (%llu MiB); blocks of %llu bytes\n", want,
         nv[want].name.c_str(), nv[want].bus, type, mp.memoryTypes[type].heapIndex,
         (unsigned long long)(mp.memoryHeaps[mp.memoryTypes[type].heapIndex].size >> 20),
         (unsigned long long)block);

  float prio = 1;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = 0;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  VkDevice dev;
  VK(vkCreateDevice(pd, &dci, nullptr, &dev));

  long base = nvrm_lines(), prev = base;
  uint64_t total = 0, bad = 0;
  std::vector<VkDeviceMemory> mems;
  for (int b = 0; b < blocks; b++) {
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = block;
    mai.memoryTypeIndex = type;
    VkDeviceMemory mem;
    VkResult r = vkAllocateMemory(dev, &mai, nullptr, &mem);
    if (r != VK_SUCCESS) { printf("block %d: vkAllocateMemory VkResult %d, stop\n", b, (int)r); break; }
    long afterAlloc = nvrm_lines();
    void* ptr = nullptr;
    r = vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, &ptr);
    long afterMap = nvrm_lines();
    if (r != VK_SUCCESS) { printf("block %d: vkMapMemory VkResult %d, stop\n", b, (int)r); vkFreeMemory(dev, mem, nullptr); break; }
    mems.push_back(mem);
    volatile uint32_t* w = (volatile uint32_t*)ptr;
    const uint64_t words = block / 4, step = 1024;  // one word per 4 KiB page
    for (uint64_t i = 0; i < words; i += step) w[i] = (uint32_t)(i * 2654435761u) ^ (uint32_t)b;
    uint64_t wrong = 0;
    for (uint64_t i = 0; i < words; i += step) wrong += w[i] != ((uint32_t)(i * 2654435761u) ^ (uint32_t)b);
    bad += wrong;
    total += block;
    printf("block %2d: mapped total %6.2f GiB | NVRM +%ld at allocate, +%ld at map | pages wrong %llu\n", b,
           total / 1073741824.0, afterAlloc - prev, afterMap - afterAlloc, (unsigned long long)wrong);
    fflush(stdout);
    prev = afterMap;
  }
  printf("done: %zu blocks, %.2f GiB mapped, NVRM lines +%ld, pages wrong %llu\n", mems.size(), total / 1073741824.0,
         nvrm_lines() - base, (unsigned long long)bad);
  for (auto m : mems) { vkUnmapMemory(dev, m); vkFreeMemory(dev, m, nullptr); }
  vkDestroyDevice(dev, nullptr);
  vkDestroyInstance(inst, nullptr);
  return bad ? 3 : 0;
}
