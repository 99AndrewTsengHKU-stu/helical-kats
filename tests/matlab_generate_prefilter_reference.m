% Generate a compact ground-truth output from the original Wang Wei MATLAB
% functions for cross-language validation of wangwei/cupy_recon.py.

addpath('D:/Github/wangwei-katsevich/helical_curve');

A = 12;
W = 7;
T = 9;
sin_full = reshape(single(sin(single(1:A*W*T) * 0.01)), [A, W, T]);
DSD = 180;
DSO = 100;
h = 3.5;
alpha = single(linspace(-0.18, 0.18, A));
w = single(linspace(-10, 10, W));
phi = single(linspace(-1.2, 1.2, 19) + 0.013);
da = alpha(2) - alpha(1);
dt = single(0.05);

g_filt = prefilter_sinogram( ...
    sin_full, DSD, h, DSO, alpha, w, phi, da, dt, T, 0);

save('C:/Temp/wangwei_prefilter_matlab_ref.mat', ...
    'sin_full', 'g_filt', 'DSD', 'DSO', 'h', ...
    'alpha', 'w', 'phi', 'da', 'dt', '-v7');

rmpath('D:/Github/wangwei-katsevich/helical_curve');
