#!/usr/bin/env bash
set -euo pipefail

IN_DIR="${1:?Usage: $0 <DICOM_INPUT_DIR>}"
WORK_DIR="${IN_DIR%/}/_split_by_acq"
OUT_DIR="${IN_DIR%/}/nii_by_acq"

command -v dcmdump >/dev/null 2>&1 || { echo "ERROR: dcmdump not found (install dcmtk)."; exit 1; }
command -v dcm2niix >/dev/null 2>&1 || { echo "ERROR: dcm2niix not found."; exit 1; }

mkdir -p "$WORK_DIR" "$OUT_DIR"

get_val() { dcmdump +P "$1" "$2" 2>/dev/null | sed -n 's/.*\[\(.*\)\].*/\1/p' | head -n 1; }

find "$IN_DIR" -type f -iname "*.dcm" -print0 |
while IFS= read -r -d '' f; do
  uid="$(get_val 0020,000e "$f")"
  acq="$(get_val 0020,0012 "$f")"
  [ -n "$uid" ] || continue
  [ -n "$acq" ] || acq="NA"
  dest="$WORK_DIR/${uid}/acq_${acq}"
  mkdir -p "$dest"
  cp -n "$f" "$dest/"
done

# Convert each acquisition separately
for acq_dir in "$WORK_DIR"/*/acq_*; do
  [ -d "$acq_dir" ] || continue
  acq_name="$(basename "$acq_dir")"
  uid_name="$(basename "$(dirname "$acq_dir")")"
  out="${OUT_DIR}/${uid_name}/${acq_name}"
  mkdir -p "$out"

  dcm2niix -o "$out" -z y -b y -ba y -m n -f "%p_%t_%s_a%a_e%e" "$acq_dir"
done

echo "Done: $OUT_DIR"
