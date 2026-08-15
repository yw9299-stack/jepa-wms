#!/usr/bin/env bash
set -euo pipefail

# Canonical task-specific LEWM-step-matched launcher.  The historical
# stablewm_task_5pass_autodl.sh filename remains as the implementation entry
# point so old audit references keep resolving, but its schedule is no longer
# uniformly five passes.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/stablewm_task_5pass_autodl.sh" "$@"
