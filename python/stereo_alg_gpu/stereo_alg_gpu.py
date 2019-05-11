import numpy as np
from numba import cuda, jit
from adapt_weight import adapt_weight
from block_matching import sad, ssd, sxd, nssd, multi_window


@cuda.jit('void(float32[:, :, :], float32[:, :])')
def argmin_disp(cost_maps_gpu, disp_mat_gpu):
    r, c = cuda.grid(2)
    ndisp, rows, cols = cost_maps_gpu.shape
    if r < rows and c < cols:
        min_cost = np.inf
        min_d = 0.0
        for d in range(ndisp):
            cost = cost_maps_gpu[d, r, c]
            if cost < min_cost:
                min_cost = cost
                min_d = d
        disp_mat_gpu[r, c] = min_d


def cost_to_disp(cost_maps_gpu):
    _, rows, cols = cost_maps_gpu.shape
    disp_mat = np.zeros([rows, cols], dtype='float32')
    disp_mat_gpu = cuda.to_device(disp_mat)

    argmin_disp[(30, 30), (30, 30)](cost_maps_gpu, disp_mat_gpu)

    return disp_mat_gpu
