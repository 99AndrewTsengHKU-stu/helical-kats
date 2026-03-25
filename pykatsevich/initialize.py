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

def create_configuration(
    scan_geometry : dict,
    astra_volume_geometry : dict,
    katsevich_options: dict={}
):
    """
    Create the helical CT configuration from the corresponding projection and volume geometries.

    Parameters:
    ===========
    scan_geometry : dict
        Dictionary with the following keys:["SOD", "SDD", "detector", "helix"].
        Keys in "detector": ["detector psize", "detector rows", "detector cols"].
        Keys in "helix": ["angles_count", "pitch_mm_rad"].
        Optional key in "helix": "angles_range".
    astra_volume_geometry : dict
        Dictionary with the ASTRA volume geometry. See astra.create_vol_geom(...) for details.

    katsevich_options: dict
        Dictionary with options for Katsevich's filtering.
        Supported keys: ["detector_rebin_rows"]
    Returns:
    ========
    helical_conf : dict
        Helical configuration.
    """

    # angles_helical = np.linspace(
    #     scan_geometry["angle_start"],
    #     scan_geometry["angle_end"],
    #     scan_geometry["angles_count"],
    #     endpoint=False
    # )
    # ang_step = angles_helical[1]-angles_helical[0]
    # magnification = scan_geometry["SDD"] / scan_geometry["SOD"]
    # h_stride = ang_step*(scan_geometry["pitch"] * scan_geometry["detector psize"] * scan_geometry["detector rows"] / (2*np.pi)) / magnification
    
    # Configure specific geometry parameters useful for Katsevish filter:
    helical_conf = {}

    helical_conf['total_projs'] = scan_geometry['helix']['angles_count']

    helical_conf['x_max'] = astra_volume_geometry['option']['WindowMaxX']
    helical_conf['x_min'] = astra_volume_geometry['option']['WindowMinX']
    helical_conf['y_max'] = astra_volume_geometry['option']['WindowMaxY']
    helical_conf['y_min'] = astra_volume_geometry['option']['WindowMinY']
    helical_conf['z_max'] = astra_volume_geometry['option']['WindowMaxZ']
    helical_conf['z_min'] = astra_volume_geometry['option']['WindowMinZ']
    # print(astra_volume_geometry)
    helical_conf['x_voxels'] = astra_volume_geometry['GridColCount']
    helical_conf['y_voxels'] = astra_volume_geometry['GridRowCount']
    helical_conf['z_voxels'] = astra_volume_geometry['GridSliceCount']

    helical_conf['x_len'] = np.float32(helical_conf['x_max'] - helical_conf['x_min'])
    helical_conf['y_len'] = np.float32(helical_conf['y_max'] - helical_conf['y_min'])
    helical_conf['z_len'] = np.float32(helical_conf['z_max'] - helical_conf['z_min'])
    helical_conf['delta_x'] = np.float32(helical_conf['x_len'] / (helical_conf['x_voxels'])) # np.float32(helical_conf['x_len'] / (helical_conf['x_voxels'] - 1))
    helical_conf['delta_y'] = np.float32(helical_conf['y_len'] / (helical_conf['y_voxels'])) # np.float32(helical_conf['y_len'] / (helical_conf['y_voxels'] - 1))
    helical_conf['delta_z'] = np.float32(helical_conf['z_len'] / max(helical_conf['z_voxels'], 1 )) # np.float32(helical_conf['z_len'] / max(helical_conf['z_voxels'] - 1, 1))

    helical_conf['detector rows'] = scan_geometry['detector']['detector rows']
    helical_conf['detector cols'] = scan_geometry['detector']['detector cols']

    # helical_conf['progress_per_radian'] = h_stride  / ang_step # mm per radian
    helical_conf['progress_per_radian'] = scan_geometry['helix']['pitch_mm_rad']
    helical_conf['progress_per_turn']   = helical_conf['progress_per_radian'] * 2*np.pi

    helical_conf['scan_diameter'] = scan_geometry["SDD"] # scan_geometry["SOD"] + scan_geometry["SDD"]
    helical_conf['scan_radius'] = scan_geometry["SOD"] # Used to be (which is incorrect): 0.5 * helical_conf['scan_diameter']
    
    if "angles_range" in scan_geometry['helix'].keys():
        helical_conf['s_len'] = scan_geometry['helix']['angles_range']
        helical_conf['s_min'] = -helical_conf['s_len'] * 0.5
        helical_conf['s_max'] =  helical_conf['s_len'] * 0.5
    else:
        helical_conf['s_min'] = -np.pi + helical_conf['z_min'] / helical_conf['progress_per_radian']
        helical_conf['s_max'] =  np.pi + helical_conf['z_max'] / helical_conf['progress_per_radian']
        helical_conf['s_len'] = helical_conf['s_max'] - helical_conf['s_min']

    helical_conf['projs_per_turn'] = helical_conf['total_projs'] / helical_conf['s_len'] * 2*np.pi

    helical_conf['delta_s'] = 2*np.pi / helical_conf['projs_per_turn'] # Turn in radians per one projection

    # print(f"projs_per_turn = {helical_conf['projs_per_turn']},\
    #         progress_per_radian = {helical_conf['progress_per_radian']}, progress_per_turn={helical_conf['progress_per_turn']}, delta_s={helical_conf['delta_s']},\
    #         s_len={helical_conf['s_len']}")

    helical_conf['source_pos'] = helical_conf['s_min'] + helical_conf['delta_s'] * (np.arange(helical_conf['total_projs'], dtype=np.float32) + 0.5 )

    # print(f"s_min = {helical_conf['s_min']}, s_max = {helical_conf['s_max']}")
    # print(f"helical_conf['source_pos'] = {helical_conf['source_pos']}")

    # Use separate pixel sizes if available; fall back to averaged 'detector psize'
    psize_cols = scan_geometry['detector'].get('detector psize cols',
                     scan_geometry['detector']['detector psize'])
    psize_rows = scan_geometry['detector'].get('detector psize rows',
                     scan_geometry['detector']['detector psize'])
    helical_conf['pixel_span'] = psize_cols       # fan (horizontal) direction
    helical_conf['pixel_height'] = psize_rows     # z (vertical) direction
    helical_conf['detector_pixel_width'] = psize_cols
    helical_conf['detector_pixel_height'] = psize_rows
    helical_conf['detector_pixel_span'] = psize_cols

    helical_conf['detector_rebin_rows'] = katsevich_options.get('detector_rebin_rows', 64) # The default value is 64
    # print(f"detector_rebin_rows = {helical_conf['detector_rebin_rows']}")

    helical_conf['fov_diameter'] = max(helical_conf['x_len'], helical_conf['y_len'])
    helical_conf['fov_radius'] = 0.5 * helical_conf['fov_diameter']
    helical_conf['half_fan_angle'] = np.arcsin(helical_conf['fov_radius'] / helical_conf['scan_radius'])
    
    helical_conf['detector_rebin_rows_height'] = (np.pi + 2* helical_conf['half_fan_angle']) / (helical_conf['detector_rebin_rows'] - 1)

    col_offset = scan_geometry['detector'].get('detector_col_offset', 0.0)
    row_offset = scan_geometry['detector'].get('detector_row_offset', 0.0)

    helical_conf['col_coords'] = psize_cols * (np.arange(scan_geometry['detector']['detector cols'] + 1, dtype=np.float32)
              - 0.5*(scan_geometry['detector']['detector cols'] - 1)
              - col_offset
              ) # EXTENDED coordinates, centered on actual detector center

    helical_conf['row_coords'] = psize_rows * (np.arange(scan_geometry['detector']['detector rows'] + 1, dtype=np.float32)
              - 0.5*(scan_geometry['detector']['detector rows'] - 1)
              - row_offset
              ) # EXTENDED coordinates, centered on actual detector center

    helical_conf['detector_col_offset'] = col_offset
    helical_conf['detector_row_offset'] = row_offset

    rebin_coords =  -np.pi / 2 - helical_conf['half_fan_angle'] + helical_conf['detector_rebin_rows_height'] * np.arange(helical_conf['detector_rebin_rows'], dtype=np.float32)

    rebin_scale = ( helical_conf['scan_diameter'] * helical_conf['progress_per_turn']) / (2 * np.pi * helical_conf['scan_radius'] )

    # Simplified version of original scale expression:
    # rebin_scale = 2 * helical_conf['progress_per_radian']

    # skip last column from differentiation
    col_coords = helical_conf['col_coords'][:-1]

    helical_conf['fwd_rebin_row'] = np.zeros( (helical_conf['detector_rebin_rows'], helical_conf['detector cols']) )
    for col in range(helical_conf['detector cols']):
        row = rebin_scale * (rebin_coords + rebin_coords
                            / np.tan(rebin_coords) * (col_coords[col]
                            / helical_conf['scan_diameter']))
        helical_conf['fwd_rebin_row'][:, col] = row

    # Hilbert filter:
    helical_conf['kernel_radius'] = helical_conf['detector cols'] - 1
    helical_conf['kernel_width'] = 1 + 2 * helical_conf['kernel_radius']
    helical_conf['proj_filter_width'] = helical_conf['detector cols']

    # Tam-Danialsson boundaries, w_top and w_bottom, from Noo et al., Eq. (78):
    proj_row_mins = -helical_conf['progress_per_turn'] / (2*np.pi*helical_conf["scan_radius"]*helical_conf["scan_diameter"]) * (helical_conf['col_coords']**2 + helical_conf['scan_diameter']**2) * (np.pi/2 + np.arctan(helical_conf['col_coords']/helical_conf['scan_diameter']))
    proj_row_maxs =  helical_conf['progress_per_turn'] / (2*np.pi*helical_conf["scan_radius"]*helical_conf["scan_diameter"]) * (helical_conf['col_coords']**2 + helical_conf['scan_diameter']**2) * (np.pi/2 - np.arctan(helical_conf['col_coords']/helical_conf['scan_diameter']))

    helical_conf['proj_row_mins'] = proj_row_mins
    helical_conf['proj_row_maxs'] = proj_row_maxs

    helical_conf['T-D smoothing'] = 0.025

    rebin_row = np.zeros(shape=(helical_conf['detector rows'], helical_conf['detector cols']), dtype=np.int32)
    
    pos_start = int(round(0.5 * (helical_conf['detector cols'] - 1) + col_offset))
    
    # Prepare helper arrays for reverse height rebinning:
    fracs_0 = np.zeros(shape=(helical_conf['detector rows'], helical_conf['detector cols']))
    fracs_1 = np.ones (shape=(helical_conf['detector rows'], helical_conf['detector cols']))
    
    for row in range(helical_conf['detector rows']):

        for col in range(pos_start, helical_conf['detector cols']):
            rebin_row[row, col] = 0
            for rebin in range(helical_conf["detector_rebin_rows"] - 1):
                if (helical_conf['row_coords'][row] >= helical_conf['fwd_rebin_row'][rebin, col]) \
                    and (helical_conf['row_coords'][row] <= helical_conf['fwd_rebin_row'][rebin + 1, col]):
                    rebin_row[row, col] = rebin
                    break

            fracs_0[row, col] = (helical_conf['row_coords'][row] - helical_conf['fwd_rebin_row'][rebin_row[row, col], col]) \
                / (helical_conf['fwd_rebin_row'][rebin_row[row, col] + 1, col] - helical_conf['fwd_rebin_row'][rebin_row[row, col], col])

        for col in range(pos_start):
            rebin_row[row, col] = 1
            for rebin in range(helical_conf['detector_rebin_rows'] - 1, 0, -1):
                if (helical_conf['row_coords'][row] >= helical_conf['fwd_rebin_row'][rebin - 1, col]) \
                    and (helical_conf['row_coords'][row] <= helical_conf['fwd_rebin_row'][rebin, col]):
                        rebin_row[row, col] = rebin
                        break
            fracs_0[row, col] = (helical_conf['row_coords'][row] - helical_conf['fwd_rebin_row'][rebin_row[row, col] - 1, col]) \
                / (helical_conf['fwd_rebin_row'][rebin_row[row, col], col] - helical_conf['fwd_rebin_row'][rebin_row[row, col] - 1, col])

    fracs_1 -= fracs_0
    
    helical_conf['rebin_row'] = rebin_row
    helical_conf['rebin_fracs_0'] = fracs_0
    helical_conf['rebin_fracs_1'] = fracs_1

    return helical_conf