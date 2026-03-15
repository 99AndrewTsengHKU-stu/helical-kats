# Pykatsevich

## Introduction

A Python package with the implementation of Katsevich algorithm for helical CT.
A good algorithm description is given in [F. Noo et al, Exact helical reconstruction using native cone-beam geometries](https://iopscience.iop.org/article/10.1088/0031-9155/48/23/001).
Filtering steps run on CPU.
Backprojection runs on a GPU, thanks to the `BP3D_CUDA` algorithm from the [ASTRA toolbox](https://github.com/astra-toolbox/astra-toolbox).

## Dependencies
`astra` >= 2, `numpy`, `cupy`, `scipy`.

## Installation

Clone the repository, then run:
```
cd pykatsevich
make install
```
This will run `pip install . --verbose` and clean up after the installation is done.
To uninstall the package, run `make uninstall`.
## Usage
A typical pipeline includes algorithm initialization (module `initialize`), preparation of projection and volume geometry (module `geometry`), filtering (module `filtering`), and reconstruction (module `reconstruct`).
For a quick example, run `python tests/t03.py`.

### Loading projection DICOM series

Use the helper in `pykatsevich.dicom` to turn a folder of projection DICOMs into the `(views, detector_rows, detector_cols)` array expected by the reconstruction code:
(install `pydicom` if it is not already available).

```python
from pykatsevich.dicom import load_dicom_projections
from pykatsevich.geometry import astra_helical_views
import astra

sino, meta = load_dicom_projections(
    r"D:\1212_High_Pitch_argparse\L067\L067_bundle_nview_div4_pitch1.5\dcm_proj"
)
scan_geom = meta["scan_geometry"]
angles = meta["angles_rad"]
z_shifts = meta["table_positions_mm"] - meta["table_positions_mm"].mean()

views = astra_helical_views(
    scan_geom["SOD"],
    scan_geom["SDD"],
    scan_geom["detector"]["detector psize"],
    angles,
    meta["pitch_mm_per_angle"],
    vertical_shifts=z_shifts,
)
proj_geom = astra.create_proj_geom(
    "cone_vec",
    scan_geom["detector"]["detector rows"],
    scan_geom["detector"]["detector cols"],
    views,
)
```

`sino` already matches the shape (4608, 64, 736) for the dataset in `dcm_proj`, and rescale slope/intercept from the headers are applied.

## Authors and acknowledgment
Pavel Paramonov and Femke Danckaers.

## License
GPL v3
