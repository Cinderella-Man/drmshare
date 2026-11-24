// Keeps one GPU out of its idle state for a while: one block spinning on the clock, no memory traffic.
// An idle GeForce card drops to P8, and its PCIe link to Gen1; writes arriving from another GPU do not wake it. In a
// narrow slot (x4) that caps what the card receives at about 1 GB/s. Run this on the RECEIVING card only while timing
// copies into it: on the sending card it would take turns with the copy and slow it down (docs/dmabuf-615.md).
// Build: nvcc -O2 keep_busy.cu -o keep_busy        Run: CUDA_DEVICE_ORDER=PCI_BUS_ID ./keep_busy <device> <seconds>
#include <cstdio>
#include <cstdlib>
__global__ void spin(long long cycles) { long long s = clock64(); while (clock64() - s < cycles) {} }
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: keep_busy <device, PCI bus order> <seconds>\n"); return 1; }
    int dev = atoi(argv[1]); double secs = atof(argv[2]);
    if (cudaSetDevice(dev) != cudaSuccess) { fprintf(stderr, "no CUDA device %d\n", dev); return 1; }
    int khz = 0; cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev);
    long long cycles = (long long)khz * 100;  // about 0.1 s per launch
    for (int i = 0; i < (int)(secs * 10); i++) spin<<<1, 32>>>(cycles);
    cudaDeviceSynchronize();
    return 0;
}
