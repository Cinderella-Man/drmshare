#!/bin/bash
# Reload all NVIDIA kernel modules with optional driver registry keys, then put the changed nvidia-drm (DRMShare) in.
# For trying a key such as RMForceStaticBar1=0 without a reboot (docs/troubleshooting.md). Keys given here apply to
# every card (NVreg_RegistryDwords) and come on top of what /etc/modprobe.d sets.
#
# usage: sudo scripts/reload-nvidia.sh <changed nvidia-drm.ko> ["Key=Value;Key2=Value2"]
#   nvidia-drm.ko: the one drv-test.sh built (default build tree: /var/tmp/drmshare-test/nvsrc/nvidia-drm.ko)
# Stops gdm and nvidia-persistenced: nothing may hold the GPUs.
set -u
DRM=$(realpath "${1:?usage: reload-nvidia.sh <nvidia-drm.ko> [keys]}")
KEYS=${2:-}
systemctl stop gdm nvidia-persistenced 2>/dev/null; sleep 2
for m in nvidia_drm nvidia_uvm nvidia_modeset nvidia; do
  if lsmod | grep -q "^$m "; then rmmod $m || { echo "rmmod $m failed; holders:"; lsof /dev/nvidia* 2>/dev/null | head; exit 1; }; fi
done
if [ -n "$KEYS" ]; then modprobe nvidia NVreg_RegistryDwords="$KEYS" || exit 1; else modprobe nvidia || exit 1; fi
modprobe nvidia_modeset && modprobe nvidia_uvm || exit 1
sleep 2
# udev or a module dependency may already have loaded the stock nvidia-drm: swap it for the changed one
if lsmod | grep -q "^nvidia_drm "; then rmmod nvidia_drm || exit 1; fi
insmod "$DRM" modeset=1 fbdev=1 peer_vidmem_dmabuf=1 || { echo "changed nvidia-drm failed to load"; exit 1; }
grep -h -E "^RegistryDwords" /proc/driver/nvidia/params
echo "peer_vidmem_dmabuf: $(cat /sys/module/nvidia_drm/parameters/peer_vidmem_dmabuf)"
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader
