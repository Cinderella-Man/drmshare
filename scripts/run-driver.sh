#!/bin/bash
# Replace the installed NVIDIA driver with NVIDIA's own .run installer <version> (open kernel modules), loaded live,
# for testing a branch Ubuntu does not package (590, 610, 620). Removes Ubuntu's NVIDIA packages first. Downloads from
# download.nvidia.com, checks NVIDIA's sha256, and unpacks the source next to the installer for drv-test.sh:
#   sudo scripts/drv-test.sh <diff> "$DRIVERS/NVIDIA-Linux-x86_64-<version>/kernel-open"
# The .run carries none of Ubuntu's kernel 7.0 fixes; if its build fails on kernel 7.0, see docs/dmabuf-drivers.md.
# Back to Ubuntu: sudo nvidia-uninstall --silent, then scripts/switch-driver.sh <branch>, reboot.
#
# usage: sudo scripts/run-driver.sh <version>      (DRIVERS: download dir, default /var/tmp/nvidia-run)
set -u
V=${1:?usage: run-driver.sh <version>, for example 610.57.04}
export DEBIAN_FRONTEND=noninteractive
DRIVERS=${DRIVERS:-/var/tmp/nvidia-run}
mkdir -p "$DRIVERS" && cd "$DRIVERS" || exit 1
R=NVIDIA-Linux-x86_64-$V.run
U=https://download.nvidia.com/XFree86/Linux-x86_64/$V
[ -s $R ] || curl -sS -f -o $R $U/$R || { echo "DOWNLOAD FAILED"; exit 1; }
curl -sS -f -o $R.sha256sum $U/$R.sha256sum && sha256sum -c $R.sha256sum || { echo "CHECKSUM FAILED"; exit 1; }
chmod +x $R
systemctl stop gdm nvidia-persistenced 2>/dev/null; sleep 3
mapfile -t OLD < <(dpkg -l | awk '/^(ii|hi)/ {print $2}' | grep nvidia | grep -E -- '-(5[0-9][0-9]|6[0-9][0-9])(-|:|$)')
if [ ${#OLD[@]} -gt 0 ]; then
  echo "removing Ubuntu packages: ${OLD[*]}"
  apt-mark unhold "${OLD[@]}" >/dev/null
  apt-get purge -y "${OLD[@]}" || { echo "PURGE FAILED"; exit 1; }
fi
for m in nvidia_drm nvidia_modeset nvidia_uvm nvidia_peermem nvidia; do lsmod | grep -q "^$m " && rmmod $m; done
lsmod | grep -E "^nvidia" && { echo "MODULES STILL LOADED"; exit 1; }
./$R --silent --kernel-module-type=open --no-questions --ui=none --no-x-check --disable-nouveau --no-cc-version-check \
  || { echo "INSTALL FAILED"; grep -i -E "error|fail" /var/log/nvidia-installer.log | tail -15; exit 1; }
modprobe nvidia_drm modeset=1 fbdev=1 || { echo "LOAD FAILED"; exit 1; }
rm -rf NVIDIA-Linux-x86_64-$V && ./$R --extract-only --target NVIDIA-Linux-x86_64-$V > /dev/null
head -1 /proc/driver/nvidia/version
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
echo "source for drv-test.sh: $DRIVERS/NVIDIA-Linux-x86_64-$V/kernel-open"
