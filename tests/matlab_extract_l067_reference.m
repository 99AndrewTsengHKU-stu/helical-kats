% Extract compact slices from the validated 2026-05-04 MATLAB baseline for
% comparison with the CuPy implementation.  Python uses zero-based indices
% [0, 224, 449], corresponding to MATLAB indices [1, 225, 450].

source_path = ...
    'D:/Github/helical-kats/out_2026-05-04/rec_L067_prefiltered_450slice.mat';
output_path = ...
    'D:/Github/helical-kats/out_cupy_2026-08-01/matlab_reference_slices.mat';

source = load(source_path, 'rf', 'hu', 'z_cor');
slice_indices = [1, 225, 450];
rf_reference = source.rf(:, :, slice_indices);
hu_reference = source.hu(:, :, slice_indices);
z_reference = source.z_cor(slice_indices);

save(output_path, 'rf_reference', 'hu_reference', ...
    'z_reference', 'slice_indices', '-v7');
