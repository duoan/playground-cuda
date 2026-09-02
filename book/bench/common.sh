#!/usr/bin/env bash
# Shared helpers for chapter bench scripts.
#
# Usage from a chapter script:
#   source "$(dirname "$0")/common.sh"
#   ncu_metrics <binary> <kernel_regex> <out_csv> <metric_list>

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
NCU="${NCU:-/usr/local/cuda-13.0/bin/ncu}"

# All ncu invocations go through sudo because the host has
# RmProfilingAdminOnly=1.  Passwordless sudo is assumed.
ncu_run() {
  sudo -n "$NCU" "$@"
}

# ncu_metrics BINARY  KERNEL_REGEX  OUT_CSV  "m1,m2,..."
#
# Filters to matching kernels, dumps metrics as csv into OUT_CSV.
ncu_metrics() {
  local binary="$1"
  local kernel="$2"
  local out="$3"
  local metrics="$4"

  ncu_run -k "$kernel" \
          --csv \
          --log-file "$out" \
          --metrics "$metrics" \
          "$binary" >/dev/null 2>&1 || true
}
