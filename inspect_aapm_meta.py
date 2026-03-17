
import pydicom
import os

def inspect_aapm():
    proj_path = r"D:\AAPM-Data\L067\L067\quarter_DICOM-CT-PD\L067_4L_100kv_quarterdose.1.00001.dcm"
    gt_path = r"D:\AAPM-Data\L067\L067\full_1mm\L067_FD_1_1.CT.0001.0001.2015.12.22.18.09.40.840353.358074219.IMA"

    print("--- Projection Metadata ---")
    if os.path.exists(proj_path):
        ds_p = pydicom.dcmread(proj_path)
        print(f"Modality: {ds_p.Modality}")
        print(f"Rows/Cols: {ds_p.Rows}x{ds_p.Columns}")
        # Check standard distances
        print(f"DistanceSourceToPatient: {ds_p.get('DistanceSourceToPatient', 'N/A')}")
        print(f"DistanceSourceToDetector: {ds_p.get('DistanceSourceToDetector', 'N/A')}")
        # Check private tags for detector size
        # AAPM-Data often uses (0x7031, 1033) for extents as we've seen
        for tag in [(0x7031, 0x1033), (0x7031, 0x1003), (0x7031, 0x1031)]:
            if tag in ds_p:
                print(f"Private Tag {tag}: {ds_p[tag].value}")
    else:
        print(f"Proj not found: {proj_path}")

    print("\n--- GT Metadata ---")
    if os.path.exists(gt_path):
        ds_g = pydicom.dcmread(gt_path)
        print(f"Rows/Cols: {ds_g.Rows}x{ds_g.Columns}")
        print(f"PixelSpacing: {ds_g.PixelSpacing}")
        print(f"SliceThickness: {ds_g.SliceThickness}")
        print(f"RescaleSlope/Intercept: {ds_g.RescaleSlope}/{ds_g.RescaleIntercept}")
        print(f"WindowCenter/Width: {ds_g.WindowCenter}/{ds_g.WindowWidth}")
        print(f"ReconstructionDiameter: {getattr(ds_g, 'ReconstructionDiameter', 'N/A')}")
    else:
        print(f"GT not found: {gt_path}")

if __name__ == "__main__":
    inspect_aapm()
