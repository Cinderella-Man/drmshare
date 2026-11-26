#!/bin/bash
# Switch Ubuntu's NVIDIA driver to another branch and hold it there, for testing DRMShare on each driver version.
# Saves the current NVIDIA package list first, so you can go back to exactly that set (apt-get install <name>=<version>).
#
# usage: sudo scripts/switch-driver.sh <branch>     then reboot
#   <branch>: 580, 595, 615, ... installs nvidia-driver-<branch>-open (open kernel modules: Turing and newer)
#             580-closed installs nvidia-driver-580, NVIDIA's closed modules (needed for Pascal and Volta)
# On Ubuntu 26.04 the real branches are 580, 595 and 615 (docs/dmabuf-drivers.md). Run it from a text console or SSH:
# the desktop goes away while the driver is replaced.
set -u
B=${1:?usage: switch-driver.sh <branch>, for example 595 or 580-closed}
export DEBIAN_FRONTEND=noninteractive
SAVE=/var/backups/nvidia-packages-$(date +%Y%m%d-%H%M%S).txt
dpkg -l | awk '/^(ii|hi)/ && $2 ~ /nvidia/ {print $2"="$3}' > "$SAVE"
echo "current NVIDIA packages saved to $SAVE; holds: $(apt-mark showhold | grep nvidia | tr '\n' ' ')"
apt-get update -qq
mapfile -t OLD < <(dpkg -l | awk '/^(ii|hi)/ {print $2}' | grep nvidia | grep -E -- '-(5[0-9][0-9]|6[0-9][0-9])(-|:|$)')
if [ ${#OLD[@]} -gt 0 ]; then
  apt-mark unhold "${OLD[@]}" >/dev/null
  apt-get purge -y "${OLD[@]}" || { echo "PURGE FAILED"; exit 1; }
fi
if [ "${B%-closed}" != "$B" ]; then PKG=nvidia-driver-${B%-closed}; N=${B%-closed}; else PKG=nvidia-driver-$B-open; N=$B; fi
apt-get install -y "$PKG" || { echo "INSTALL FAILED"; exit 1; }
mapfile -t HELD < <(dpkg -l | awk '/^(ii|hi)/ {print $2}' | grep nvidia | grep -E -- "-$N(-|:|$)")
apt-mark hold "${HELD[@]}" >/dev/null
echo "held: $(apt-mark showhold | grep nvidia | tr '\n' ' ')"
dkms status | grep nvidia
echo "reboot to load driver $B"
