#!/bin/bash
# Build Hidden GEM's nvidia-drm for the running driver and test it on two GPUs, in one go (docs/dmabuf-615.md D1-D3).
# Copies the running driver's source, applies the patches its dkms.conf lists for this kernel (as DKMS does), then the
# Hidden GEM diff, builds, checks that the stock nvidia-drm refuses the import, swaps in the changed module
# (peer_vidmem_dmabuf=1), and runs: 1 GiB each way with the receiving card kept busy, both ways at once, CUDA through
# host RAM as the baseline, a 10 x 1 GiB soak, and the kernel log.
#
# usage: sudo scripts/drv-test.sh <Hidden GEM diff> [driver source dir]
#   diff:  the 580 or 615 version (docs/dmabuf-drivers.md says which), e.g. from the submodules:
#          git -C tools/open-gpu-kernel-modules-615 diff HEAD~1 HEAD -- kernel-open/nvidia-drm > /tmp/hidden-gem.diff
#          (needs the submodule fetched with --depth 2)
#   source: default /usr/src/nvidia-<version> (Ubuntu's DKMS source); for NVIDIA's .run, its unpacked kernel-open dir
# Environment: GPU_A, GPU_B (default 0 and 1, nvidia-smi order), BIN (compiled p2p_dmabuf, keep_busy, staged_baseline;
#   default tools/), WORK (build tree and logs; default /var/tmp/hidden-gem-test).
# Stops gdm and nvidia-persistenced (nothing may hold nvidia-drm). The changed module stays loaded until the next reboot.
# Sets kernel.panic=10 and kernel.panic_on_oops=1 for this boot, so a crash reboots the machine instead of hanging it.
set -u
DIFF=$(realpath "$1")
REPO=$(cd "$(dirname "$0")/.." && pwd)
BIN=$(realpath "${BIN:-$REPO/tools}")
WORK=${WORK:-/var/tmp/hidden-gem-test}
A=${GPU_A:-0}; B=${GPU_B:-1}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
for t in p2p_dmabuf keep_busy staged_baseline; do [ -x "$BIN/$t" ] || { echo "build $BIN/$t first (see its header)"; exit 1; }; done
V=$(modinfo -F version nvidia); KV=$(uname -r)
echo "== driver $V ($(grep -o 'Open Kernel Module' /proc/driver/nvidia/version || echo closed)), kernel $KV, diff $(basename "$DIFF")"
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader
SRC=${2:-/usr/src/nvidia-$V}
mkdir -p "$WORK" && cd "$WORK" || exit 1
rm -rf nvsrc && cp -a "$SRC" nvsrc && cd nvsrc || exit 1
[ -f dkms.conf ] || : > dkms.conf
for i in $(grep -o -E '^PATCH\[[0-9]+\]' dkms.conf | grep -o -E '[0-9]+'); do
  p=$(sed -n "s/^PATCH\[$i\]=[\"']\(.*\)[\"']$/\1/p" dkms.conf)
  m=$(sed -n "s/^PATCH_MATCH\[$i\]=[\"']\(.*\)[\"']$/\1/p" dkms.conf)
  if [ -z "$m" ] || [[ $KV =~ $m ]]; then patch -p1 -s < patches/$p && echo "   distribution fix applied: $p"; fi
done
patch -p2 -s < "$DIFF" || { echo "HIDDEN GEM DIFF DID NOT APPLY"; exit 1; }
make -j"$(nproc)" SYSSRC=/lib/modules/$KV/build modules > ../build-$V.log 2>&1 || { echo "BUILD FAILED"; grep -m5 "error" ../build-$V.log; exit 1; }
echo "   built; $(modinfo -F parm nvidia-drm.ko | grep -o 'peer_vidmem_dmabuf:[^(]*' | cut -c1-60)"
cd ..
sysctl -q -w kernel.panic=10 kernel.panic_on_oops=1
echo "== stock nvidia-drm: $("$BIN"/p2p_dmabuf $B $A 64 2>&1 | grep -E 'cannot|ALL OK|FAIL' | tr '\n' ' ')"
systemctl stop gdm nvidia-persistenced 2>/dev/null; sleep 3
rmmod nvidia_drm && insmod nvsrc/nvidia-drm.ko modeset=1 fbdev=1 peer_vidmem_dmabuf=1 || { echo "module swap failed"; exit 1; }
echo "== changed nvidia-drm, option $(cat /sys/module/nvidia_drm/parameters/peer_vidmem_dmabuf)"
dmesg -C
for d in "$B $A" "$A $B"; do
  r=${d%% *}; "$BIN"/keep_busy $r 300 & KB=$!; sleep 3
  echo "-- A B = $d, 1 GiB, receiver $r kept busy: $("$BIN"/p2p_dmabuf $d 1024 2>&1 | grep -E 'exporter|copies into|reads from|4-byte|cannot|ALL OK|FAIL' | tr -s ' ' | tr '\n' ' ')"
  kill $KB 2>/dev/null; wait $KB 2>/dev/null; sleep 2
done
T=$(mktemp -d)
NO_READ=1 REPS=20 BARRIER=$T "$BIN"/p2p_dmabuf $B $A 1024 > $T/a.log 2>&1 &
NO_READ=1 REPS=20 BARRIER=$T "$BIN"/p2p_dmabuf $A $B 1024 > $T/b.log 2>&1 &
until [ -e $T/ready.$B$A ] && [ -e $T/ready.$A$B ]; do sleep 0.05; done
touch $T/go; wait
x=$(grep -o -E "copies into A: +[0-9.]+" $T/a.log | grep -o -E "[0-9.]+$"); y=$(grep -o -E "copies into A: +[0-9.]+" $T/b.log | grep -o -E "[0-9.]+$")
echo "-- both at once (1 GiB x 20 each): GPU $A into GPU $B $x + GPU $B into GPU $A $y = $(echo "$x + $y" | bc) GB/s; $(grep -h -E 'ALL OK|FAIL|WRONG' $T/a.log $T/b.log | tr '\n' ' ')"
rm -rf $T
for d in "$B $A" "$A $B"; do
  echo "-- CUDA through host RAM, A B = $d: $(timeout 300 "$BIN"/staged_baseline $d 64 2>&1 | grep -E 'peer access|cudaMemcpyPeer|4-byte' | tr -s ' ' | tr '\n' ' ')"
done
L=soak-$V.log; : > $L
for i in $(seq 1 5); do for d in "$B $A" "$A $B"; do echo "== $d run $i" >> $L; timeout 600 "$BIN"/p2p_dmabuf $d 1024 >> $L 2>&1; echo "exit $?" >> $L; done; done
echo "-- soak 10 x 1 GiB: runs $(grep -c '^exit' $L), ok $(grep -c '^exit 0' $L), non-zero wrong $(grep -E 'wrong words' $L | grep -v -c '(0 wrong words)')"
echo "== kernel log: $(dmesg | grep -c NVRM) NVRM lines, $(dmesg | grep -c rmapiLockIsOwner) lock assertions, $(dmesg | grep -c 'import dma_buf') import errors"
echo "== the changed nvidia-drm stays loaded until the next reboot; build and logs in $WORK"
