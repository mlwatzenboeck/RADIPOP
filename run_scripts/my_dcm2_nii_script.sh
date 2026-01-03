#!/usr/bin/env bash
set -euo pipefail

# Usage check
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <DICOM_INPUT_DIR>" >&2
  exit 1
fi

# Input DICOM folder (may contain multiple series / acquisition times)
IN_DIR="$1"

# Basic sanity check
if [ ! -d "$IN_DIR" ]; then
  echo "ERROR: Input directory does not exist: $IN_DIR" >&2
  exit 1
fi

# Output folder (created next to the input directory)
OUT_DIR="${IN_DIR%/}/nii"

# Ensure dcm2niix is available
command -v dcm2niix >/dev/null 2>&1 || {
  echo "ERROR: dcm2niix not found in PATH." >&2
  exit 1
}

mkdir -p "$OUT_DIR"

# Run conversion
# - recursive input handling
# - no merging across differing acquisition attributes
# - gzip-compressed NIfTI
# - JSON sidecars
# - filename template robust to multiple series / times
dcm2niix \
  -o "$OUT_DIR" \
  -z y \
  -b y \
  -ba y \
  -m n \
  -f "%p_%t_%3s" \
  "$IN_DIR"
