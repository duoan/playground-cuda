#!/usr/bin/env bash
# Wrapper that just invokes the Python runner.  Kept for backward compatibility;
# see scripts/run_ncu_sweep.py for the actual logic.
set -euo pipefail
exec python3 "$(dirname "$0")/run_ncu_sweep.py" "$@"
