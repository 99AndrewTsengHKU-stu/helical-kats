# -----------------------------------------------------------------------
# This file is part of Pykatsevich distribution (https://github.com/astra-toolbox/helical-kats).
# Copyright (c) 2024 imec-Vision Lab, University of Antwerp.
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------

import numpy as np

def astra_helical_views(
        SOD: float,
        SDD: float,
        pixel_size: float,
        angles: np.ndarray,
        pitch_per_angle:float,
        vertical_shifts: np.ndarray | None = None,
        pixel_size_col: float | None = None,
        pixel_size_row: float | None = None,
    ):
    """
    Generate ASTRA views from the helix description.

    Parameters:
    ===========
    SOD : float
        Source-objet distance.
    SDD : float
        Source-detector distance.
    pixel_size : float
        Size of detetor pixels (used for both col and row if separate sizes not given).
    angles : np.ndarray
        Array of projection angles, counter-clockwise rotation around Z.
        Each value is the angle between the X-axis and projection direction.
    pitch_per_angle : float
        Vertical pitch size per every projection angle in chosen units, e.g., mm.
    vertical_shifts : np.ndarray, optional
        Explicit z-shifts per view. When provided, this overrides the linear
        spacing defined by ``pitch_per_angle``.
    pixel_size_col : float, optional
        Detector pixel size in the column (fan/horizontal) direction.
    pixel_size_row : float, optional
        Detector pixel size in the row (z/vertical) direction.

    Return:
    =======
        Array of 12-element vectors, each vector describing a single projection view.
    """
    rot = lambda x, theta: [x[0]*np.cos(theta)-x[1]*np.sin(theta),x[0]*np.sin(theta)+x[1]*np.cos(theta),x[2]]

    angles = np.asarray(angles, dtype=np.float32)

    if vertical_shifts is None:
        vertical_shift = np.linspace(
            -pitch_per_angle*angles.shape[0]*0.5,
            pitch_per_angle*angles.shape[0]*0.5,
            angles.shape[0],
            dtype=np.float32
        )
    else:
        vertical_shift = np.asarray(vertical_shifts, dtype=np.float32)
        if vertical_shift.shape[0] != angles.shape[0]:
            raise ValueError("vertical_shifts must have the same length as angles")

    views_list = []

    ps_col = pixel_size_col if pixel_size_col is not None else pixel_size
    ps_row = pixel_size_row if pixel_size_row is not None else pixel_size
    start_view = [SOD, 0, 0, -(SDD - SOD), 0, 0, 0, ps_col, 0, 0, 0, ps_row]

    for i, aa in enumerate(angles):
        views_list.append(np.concatenate((
            rot(start_view[0:3], aa) + np.array([0, 0, vertical_shift[i]], dtype=np.float32),
            rot(start_view[3:6], aa) + np.array([0, 0, vertical_shift[i]], dtype=np.float32),
            rot(start_view[6:9], aa),
            rot(start_view[9:12],aa)
        )))

    views_array = np.asarray(views_list)
    return views_array
