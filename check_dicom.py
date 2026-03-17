import os
import sys
import pydicom
import struct

def _decode_float32_tag(ds, tag, count=1):
    if tag not in ds:
        return None
    raw = bytes(ds[tag].value)
    needed = 4 * count
    if len(raw) < needed:
        return None
    values = struct.unpack("<" + "f" * count, raw[:needed])
    return values if count > 1 else values[0]

def main():
    dicom_dir = r"D:\1212_High_Pitch_argparse\C001\C001_bundle_nview_div4_pitch0.6\dcm_proj"
    if not os.path.exists(dicom_dir):
        return

    files = [f for f in os.listdir(dicom_dir) if f.lower().endswith('.dcm')]
    if not files:
        return

    first_file = os.path.join(dicom_dir, files[0])
    ds = pydicom.dcmread(first_file)

    print("--- EXTENDED TAG CHECK ---")
    
    # Detector Extents (Private)
    extents = _decode_float32_tag(ds, (0x7031, 0x1033), count=2)
    print(f"Detector Extents (0x7031, 0x1033): {extents}")

    # Geometry
    sod = _decode_float32_tag(ds, (0x7031, 0x1003)) # Fallback private
    sdd = _decode_float32_tag(ds, (0x7031, 0x1031)) # Fallback private
    print(f"SOD (Private): {sod}")
    print(f"SDD (Private): {sdd}")
    
    std_sod = ds.get((0x0018, 0x9402))
    std_sdd = ds.get((0x0018, 0x1110))
    print(f"SOD (Std): {std_sod}")
    print(f"SDD (Std): {std_sdd}")

    # Pitch factor
    pitch = ds.get((0x0018, 0x9311))
    print(f"Spiral Pitch Factor: {pitch}")

if __name__ == "__main__":
    main()
