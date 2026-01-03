# Requires dcmtk (dcmdump). If you don't have it: sudo apt-get install dcmtk
set -euo pipefail

IN_DIR="/home/clemens/data/RADIPOP_EXTRA/working_env/FINAL.11"

find "$IN_DIR" -type f -iname "*.dcm" -print0 \
| xargs -0 -n 1 dcmdump +P 0020,000e +P 0020,0011 +P 0020,0012 +P 0008,0032 +P 0018,0081 +P 0018,0050 +P 0008,0008 2>/dev/null \
| awk '
  /SeriesInstanceUID/ {uid=$0}
  /SeriesNumber/      {ser=$0}
  /AcquisitionNumber/ {acq=$0}
  /AcquisitionTime/   {tim=$0}
  /EchoTime/          {te=$0}
  /SliceThickness/    {st=$0}
  /ImageType/         {it=$0; print uid " | " ser " | " acq " | " tim " | " te " | " st " | " it}
' \
| sort -u | head -n 200
