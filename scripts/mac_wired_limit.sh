#!/usr/bin/env bash
# Let the Apple Silicon GPU use more unified memory (macOS default is ~75% of RAM).
# Needed for ~20B QLoRA on a 32 GB Mac. Resets on reboot.
#   sudo ./scripts/mac_wired_limit.sh            # default: RAM minus 6 GB for macOS
#   sudo ./scripts/mac_wired_limit.sh 26624      # explicit value in MB
#   sudo ./scripts/mac_wired_limit.sh reset
set -euo pipefail
if [[ "$(uname)" != "Darwin" ]]; then echo "macOS only"; exit 1; fi
if [[ "${1:-}" == "reset" ]]; then sysctl iogpu.wired_limit_mb=0; exit 0; fi
total_mb=$(( $(sysctl -n hw.memsize) / 1024 / 1024 ))
limit_mb=${1:-$(( total_mb - 6144 ))}
if (( limit_mb > total_mb - 4096 )); then
  echo "Refusing: leave at least 4 GB for macOS (total ${total_mb} MB)"; exit 1
fi
sysctl iogpu.wired_limit_mb="${limit_mb}"
echo "GPU wired limit set to ${limit_mb} MB of ${total_mb} MB (until reboot)."
