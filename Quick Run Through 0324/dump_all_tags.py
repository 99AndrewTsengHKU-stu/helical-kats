"""Dump all private tags from a DICOM projection file to check for misinterpretation."""
import struct
from pathlib import Path
import pydicom
import numpy as np

DICOM_DIR = Path(r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD")
paths = sorted(DICOM_DIR.glob("*.dcm"))

# Read first and second file for comparison
for i, p in enumerate(paths[:2]):
    ds = pydicom.dcmread(p, stop_before_pixels=True)
    print(f"\n{'='*60}")
    print(f"File {i}: {p.name}")
    print(f"{'='*60}")

    # All tags
    for elem in ds:
        tag = elem.tag
        # Skip pixel data and very long values
        if tag == (0x7FE0, 0x0010):
            continue

        val_repr = str(elem.value)
        if len(val_repr) > 200:
            val_repr = val_repr[:200] + "..."

        # Try to decode private float tags
        extra = ""
        if tag.group & 1:  # odd group = private
            try:
                raw = bytes(elem.value)
                if len(raw) == 4:
                    f32 = struct.unpack("<f", raw)[0]
                    extra = f"  [as float32: {f32}]"
                elif len(raw) == 8:
                    f32x2 = struct.unpack("<ff", raw)
                    f64 = struct.unpack("<d", raw)[0]
                    extra = f"  [as 2xfloat32: {f32x2}] [as float64: {f64}]"
                elif len(raw) == 2:
                    u16 = struct.unpack("<H", raw)[0]
                    i16 = struct.unpack("<h", raw)[0]
                    extra = f"  [as uint16: {u16}] [as int16: {i16}]"
            except:
                pass

        print(f"  ({tag.group:04X},{tag.element:04X}) {elem.VR:2s} {elem.keyword or elem.name:40s} = {val_repr}{extra}")

# Also check what tags we currently USE vs what's available
print(f"\n{'='*60}")
print("TAGS CURRENTLY USED IN dicom.py:")
print(f"{'='*60}")
used_tags = {
    "(7031,1001)": "ANGLE_TAG - source angle (rad)",
    "(0018,9327)": "TABLE_POSITION_TAG - table z (mm)",
    "(7031,1002)": "DETECTOR_AXIAL_POSITION_TAG - fallback z",
    "(0018,9402)": "SOD_TAG - source-object distance",
    "(0018,1110)": "SDD_TAG - source-detector distance",
    "(7031,1033)": "DETECTOR_CENTRAL_ELEMENT_TAG - center pixel index",
    "(7029,1002)": "DETECTOR_COL_SPACING_TAG - col spacing at detector",
    "(7029,1006)": "DETECTOR_ROW_SPACING_TAG - row spacing at detector",
    "(7029,100B)": "DETECTOR_SHAPE_TAG - CYLINDRICAL/FLAT",
    "(7029,1010)": "n_rows fallback",
    "(0018,9311)": "Spiral Pitch Factor",
}
for tag, desc in used_tags.items():
    print(f"  {tag}: {desc}")

# Check for private tags we're NOT using
print(f"\n{'='*60}")
print("PRIVATE TAGS NOT CURRENTLY USED:")
print(f"{'='*60}")
used_raw = {(0x7031,0x1001), (0x0018,0x9327), (0x7031,0x1002),
            (0x0018,0x9402), (0x0018,0x1110), (0x7031,0x1033),
            (0x7029,0x1002), (0x7029,0x1006), (0x7029,0x100B),
            (0x7029,0x1010), (0x0018,0x9311)}

ds = pydicom.dcmread(paths[0], stop_before_pixels=True)
for elem in ds:
    tag = (elem.tag.group, elem.tag.element)
    if tag[0] & 1 and tag not in used_raw:  # private, not used
        extra = ""
        try:
            raw = bytes(elem.value)
            if len(raw) == 4:
                f32 = struct.unpack("<f", raw)[0]
                extra = f"  [float32: {f32}]"
            elif len(raw) == 8:
                f32x2 = struct.unpack("<ff", raw)
                extra = f"  [2xf32: {f32x2}]"
            elif len(raw) == 2:
                u16 = struct.unpack("<H", raw)[0]
                extra = f"  [uint16: {u16}]"
        except:
            pass
        val_repr = str(elem.value)[:100]
        print(f"  ({tag[0]:04X},{tag[1]:04X}) {elem.VR:2s} = {val_repr}{extra}")
