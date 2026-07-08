#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "setup_svc_wsl.sh is kept for compatibility; using setup_svc_ubuntu.sh."
exec "${SCRIPT_DIR}/setup_svc_ubuntu.sh" "$@"
