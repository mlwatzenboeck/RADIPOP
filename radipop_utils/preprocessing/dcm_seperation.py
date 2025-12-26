from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import tqdm

@dataclass(frozen=True)
class DicomMeta:
    path: Path

    # identity / grouping
    patient_name: str                       # (0010,0010)
    acquisition_time: str                   # (0008,0032)

    series_instance_uid: Optional[str] = None   # (0020,000E)
    series_number: Optional[str] = None         # (0020,0011)
    acquisition_number: Optional[str] = None    # (0020,0012)
    instance_number: Optional[int] = None       # (0020,0013)

    # phase / protocol hints (optional)
    series_description: Optional[str] = None    # (0008,103E)
    protocol_name: Optional[str] = None         # (0018,1030)
    image_type: Optional[str] = None            # (0008,0008)

    # geometry (optional)
    slice_thickness_mm: Optional[float] = None  # (0018,0050)
    spacing_between_slices_mm: Optional[float] = None  # (0018,0088)


def _is_valid_filename(name: str) -> bool:
    return not (name.startswith(".") or name.startswith("_"))


def _sorted_files(folder: Path, suffix: str) -> List[Path]:
    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == suffix and _is_valid_filename(p.name)
    ]
    return sorted(files, key=lambda p: p.name)


def _try_read_with_pydicom(dcm_path: Path) -> Optional[DicomMeta]:
    try:
        import pydicom  # type: ignore
    except Exception:
        return None

    try:
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
    except Exception:
        return None

    def get_str(tag_name: str, default: str = "") -> str:
        v = getattr(ds, tag_name, default)
        return "" if v is None else str(v)

    def get_float(tag_name: str) -> Optional[float]:
        v = getattr(ds, tag_name, None)
        if v is None or v == "":
            return None
        try:
            return float(v)
        except Exception:
            return None

    inst_no_val = getattr(ds, "InstanceNumber", None)
    inst_no = None
    try:
        if inst_no_val not in (None, ""):
            inst_no = int(inst_no_val)
    except Exception:
        inst_no = None

    return DicomMeta(
        path=dcm_path,
        patient_name=get_str("PatientName", ""),
        acquisition_time=get_str("AcquisitionTime", ""),
        series_instance_uid=(get_str("SeriesInstanceUID", "") or None),
        series_number=(get_str("SeriesNumber", "") or None),
        acquisition_number=(get_str("AcquisitionNumber", "") or None),
        instance_number=inst_no,
        series_description=(get_str("SeriesDescription", "") or None),
        protocol_name=(get_str("ProtocolName", "") or None),
        image_type=(get_str("ImageType", "") or None),
        slice_thickness_mm=get_float("SliceThickness"),
        spacing_between_slices_mm=get_float("SpacingBetweenSlices"),
    )


def _dcmdump_value(dcm_path: Path, tag: str) -> str:
    cmd = ["dcmdump", "+P", tag, str(dcm_path)]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
    # dcmtk prints: (gggg,eeee) VR [VALUE] # ...
    m = re.search(r"\[(.*?)\]", out)
    return m.group(1) if m else ""


def _dcmdump_float(dcm_path: Path, tag: str) -> Optional[float]:
    s = _dcmdump_value(dcm_path, tag)
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _read_with_dcmdump(dcm_path: Path) -> DicomMeta:
    # Tags:
    # PatientName            (0010,0010)
    # AcquisitionTime        (0008,0032)
    # SeriesInstanceUID      (0020,000E)
    # SeriesNumber           (0020,0011)
    # AcquisitionNumber      (0020,0012)
    # InstanceNumber         (0020,0013)
    # SeriesDescription      (0008,103E)
    # ProtocolName           (0018,1030)
    # ImageType              (0008,0008)
    # SliceThickness         (0018,0050)
    # SpacingBetweenSlices   (0018,0088)
    patient_name = _dcmdump_value(dcm_path, "0010,0010")
    acq_time = _dcmdump_value(dcm_path, "0008,0032")

    series_uid = _dcmdump_value(dcm_path, "0020,000e") or None
    series_no = _dcmdump_value(dcm_path, "0020,0011") or None
    acq_no = _dcmdump_value(dcm_path, "0020,0012") or None

    inst_no_str = _dcmdump_value(dcm_path, "0020,0013")
    inst_no = int(inst_no_str) if inst_no_str.isdigit() else None

    series_desc = _dcmdump_value(dcm_path, "0008,103e") or None
    proto_name = _dcmdump_value(dcm_path, "0018,1030") or None
    image_type = _dcmdump_value(dcm_path, "0008,0008") or None

    st = _dcmdump_float(dcm_path, "0018,0050")
    sbs = _dcmdump_float(dcm_path, "0018,0088")

    return DicomMeta(
        path=dcm_path,
        patient_name=patient_name,
        acquisition_time=acq_time,
        series_instance_uid=series_uid,
        series_number=series_no,
        acquisition_number=acq_no,
        instance_number=inst_no,
        series_description=series_desc,
        protocol_name=proto_name,
        image_type=image_type,
        slice_thickness_mm=st,
        spacing_between_slices_mm=sbs,
    )


def _read_dicom_meta(dcm_path: Path) -> DicomMeta:
    meta = _try_read_with_pydicom(dcm_path)
    if meta is not None:
        return meta

    if shutil.which("dcmdump") is None:
        raise RuntimeError(
            "Neither pydicom is importable nor dcmdump is available in PATH. "
            "Install one of them (recommended: pip install pydicom; or sudo apt-get install dcmtk)."
        )
    return _read_with_dcmdump(dcm_path)


def _sanitize_for_folder(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s or "UNKNOWN"


def _write_csv(csv_path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def separate_and_link_by_acquisition_time(folder_name: str) -> None:
    root = Path(folder_name).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    dcm_files = _sorted_files(root, ".dcm")

    if len(dcm_files) == 0:
        raise RuntimeError(f"No .dcm files found in: {root}")

    metas: List[DicomMeta] = [_read_dicom_meta(f) for f in dcm_files]

    # Assert patient name constant
    patient_names = sorted({m.patient_name for m in metas})
    if len(patient_names) != 1:
        raise AssertionError(
            f"PatientName is not constant. Found {len(patient_names)} distinct values: {patient_names}"
        )
    patient_name = patient_names[0]
    print(f"PatientName: {patient_name}")

    # AcquisitionTime sanity
    acq_times = sorted({m.acquisition_time for m in metas})
    if any(t == "" for t in acq_times):
        raise AssertionError(
            "At least one DICOM has empty AcquisitionTime (0008,0032). "
            "If this is expected, group by AcquisitionNumber or SeriesInstanceUID instead."
        )

    print(f"Unique AcquisitionTime values: {len(acq_times)}")
    for t in acq_times:
        print(f"  - {t}")
    
    # Count files per acquisition time
    by_time: Dict[str, List[int]] = {t: [] for t in acq_times}
    for i, m in enumerate(metas):
        by_time[m.acquisition_time].append(i)
    
    print("\nCount per AcquisitionTime:")
    for t in acq_times:
        count = len(by_time[t])
        print(f"  - {t}: {count} file(s)")
    
    if len(acq_times) > 5:
        raise RuntimeError(f"More than 5 acquisitions detected ({len(acq_times)}). Aborting as requested.")

    out_root = root / "by_acqtime"
    out_root.mkdir(parents=True, exist_ok=True)

    def safe_symlink(src: Path, dst: Path) -> None:
        if dst.exists() or dst.is_symlink():
            return
        rel = os.path.relpath(src, start=dst.parent)
        os.symlink(rel, dst)

    fields = [
        "index_in_sorted_lists",
        "patient_name",
        "acquisition_time",
        "series_instance_uid",
        "series_number",
        "acquisition_number",
        "instance_number",
        "series_description",
        "protocol_name",
        "image_type",
        "slice_thickness_mm",
        "spacing_between_slices_mm",
        "dcm_name",
        "dcm_src",
        "dcm_link",
    ]

    all_rows: List[dict] = []

    for acq_time in acq_times:
        idxs = by_time[acq_time]
        acq_dir = out_root / _sanitize_for_folder(acq_time)
        acq_dir.mkdir(parents=True, exist_ok=True)

        rows: List[dict] = []
        for i in idxs:
            dcm_src = dcm_files[i]

            dcm_link = acq_dir / dcm_src.name

            safe_symlink(dcm_src, dcm_link)

            m = metas[i]
            row = {
                "index_in_sorted_lists": i,
                "patient_name": m.patient_name,
                "acquisition_time": m.acquisition_time,
                "series_instance_uid": m.series_instance_uid or "",
                "series_number": m.series_number or "",
                "acquisition_number": m.acquisition_number or "",
                "instance_number": "" if m.instance_number is None else m.instance_number,
                "series_description": m.series_description or "",
                "protocol_name": m.protocol_name or "",
                "image_type": m.image_type or "",
                "slice_thickness_mm": "" if m.slice_thickness_mm is None else m.slice_thickness_mm,
                "spacing_between_slices_mm": "" if m.spacing_between_slices_mm is None else m.spacing_between_slices_mm,
                "dcm_name": dcm_src.name,
                "dcm_src": str(dcm_src),
                "dcm_link": str(dcm_link),
            }
            rows.append(row)
            all_rows.append(row)

        _write_csv(acq_dir / "metadata.csv", rows, fields)
        print(f"Wrote {len(idxs)} linked DICOM files + metadata.csv to: {acq_dir}")

    _write_csv(out_root / "metadata_all.csv", all_rows, fields)
    print(f"Wrote global CSV: {out_root / 'metadata_all.csv'}")
    print("Done.")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Separate and link DICOM files by acquisition time")
    parser.add_argument("--folder_name", type=str, help="Path to the folder containing the DICOM files", required=True)
    args = parser.parse_args()
    separate_and_link_by_acquisition_time(folder_name=args.folder_name)
    print("Done with separating and linking DICOM files by acquisition time!")