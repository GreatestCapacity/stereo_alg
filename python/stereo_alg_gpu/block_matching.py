import numpy as np
import math
from numba import cuda
from basic_alg import argmin, sub, mul, div, absolute, square, sqrt, aggregt


@cuda.jit
def xd(mat1, mat2, result, s, t):
    r, c = cuda.grid(2)
    if r < mat1.shape[0] and c < mat1.shape[1]:
        result[r, c] = s / (1 + math.exp(-(abs(mat2[r, c] - mat1[r, c])-t)/(0.14 * t)))


def sad(mat1_gpu, mat2_gpu, patch_size, maxdisp):
    rows, cols = mat1_gpu.shape
    rad = patch_size // 2
    ndisp = maxdisp + 1
    result = np.full([rows, cols], np.inf, dtype='float32')
    cost_maps = np.full([ndisp, rows, cols], np.inf, dtype='float32')

    result_gpu = cuda.to_device(result)
    cost_maps_gpu = cuda.to_device(cost_maps)

    for d in range(ndisp):
        sub[(30, 30), (30, 30)](mat1_gpu[:, d:], mat2_gpu[:, :cols-1-d], result_gpu[:, d:])
        absolute[(30, 30), (30, 30)](result_gpu[:, d:], result_gpu[:, d:])
        aggregt[(30, 30), (30, 30)](result_gpu[:, d:], cost_maps_gpu[d, :, d:], rad)

    return cost_maps_gpu


def ssd(mat1_gpu, mat2_gpu, patch_size, maxdisp):
    rows, cols = mat1_gpu.shape
    rad = patch_size // 2
    ndisp = maxdisp + 1
    result = np.full([rows, cols], np.inf, dtype='float32')
    cost_maps = np.full([ndisp, rows, cols], np.inf, dtype='float32')

    result_gpu = cuda.to_device(result)
    cost_maps_gpu = cuda.to_device(cost_maps)

    for d in range(ndisp):
        sub[(30, 30), (30, 30)](mat1_gpu[:, d:], mat2_gpu[:, :cols-1-d], result_gpu[:, d:])
        square[(30, 30), (30, 30)](result_gpu[:, d:], result_gpu[:, d:])
        aggregt[(30, 30), (30, 30)](result_gpu[:, d:], cost_maps_gpu[d, :, d:], rad)

    return cost_maps_gpu


def nssd(mat1_gpu, mat2_gpu, patch_size, maxdisp):
    rows, cols = mat1_gpu.shape
    rad = patch_size // 2
    ndisp = maxdisp + 1
    result = np.full([rows, cols], np.inf, dtype='float32')
    cost_maps = np.full([ndisp, rows, cols], np.inf, dtype='float32')

    result_gpu = cuda.to_device(result)
    cost_maps_gpu = cuda.to_device(cost_maps)

    denomi1 = np.zeros_like(result)
    denomi2 = np.zeros_like(result)
    denomi1_gpu = cuda.to_device(denomi1)
    denomi2_gpu = cuda.to_device(denomi2)

    square[(30, 30), (30, 30)](mat1_gpu, result_gpu)
    aggregt[(30, 30), (30, 30)](result_gpu, denomi1_gpu, rad)
    sqrt[(30, 30), (30, 30)](denomi1_gpu, denomi1_gpu)

    square[(30, 30), (30, 30)](mat2_gpu, result_gpu)
    aggregt[(30, 30), (30, 30)](result_gpu, denomi2_gpu, rad)
    sqrt[(30, 30), (30, 30)](denomi2_gpu, denomi2_gpu)

    for d in range(ndisp):
        sub[(30, 30), (30, 30)](mat1_gpu[:, d:], mat2_gpu[:, :cols-1-d], result_gpu[:, d:])
        square[(30, 30), (30, 30)](result_gpu[:, d:], result_gpu[:, d:])
        aggregt[(30, 30), (30, 30)](result_gpu[:, d:], cost_maps_gpu[d, :, d:], rad)
        mul[(30, 30), (30, 30)](denomi1_gpu[:, d:], denomi2_gpu[:, :cols-1-d], result_gpu[:, d:])
        div[(30, 30), (30, 30)](cost_maps_gpu[d, :, d:], result_gpu[:, d:], cost_maps_gpu[d, :, d:])

    return cost_maps_gpu


def sxd(mat1_gpu, mat2_gpu, patch_size, maxdisp, s=255, t=12.5):
    rows, cols = mat1_gpu.shape
    rad = patch_size // 2
    ndisp = maxdisp + 1
    result = np.full([rows, cols], np.inf, dtype='float32')
    cost_maps = np.full([ndisp, rows, cols], np.inf, dtype='float32')

    result_gpu = cuda.to_device(result)
    cost_maps_gpu = cuda.to_device(cost_maps)

    for d in range(ndisp):
        xd[(30, 30), (30, 30)](mat1_gpu[:, d:], mat2_gpu[:, :cols-1-d], result_gpu[:, d:], s, t)
        aggregt[(30, 30), (30, 30)](result_gpu[:, d:], cost_maps_gpu[d, :, d:], rad)

    return cost_maps_gpu


@cuda.jit
def multi_win(disp_mat, cost_maps, rad):
    r, c = cuda.grid(2)
    if r < disp_mat.shape[0] and c < disp_mat.shape[1]:
        min_cost = math.inf
        min_disp = 0.0
        for i in range(-rad, rad+1, rad):
            if disp_mat.shape[0] <= r + i or r + i < 0:
                continue
            for j in range(-rad, rad+1, rad):
                if disp_mat.shape[1] <= c + j or c + j < 0:
                    continue

                d = disp_mat[r + i, c + j]
                wt_c = cost_maps[int(d), r, c]
                if wt_c < min_cost:
                    min_cost = wt_c
                    min_disp = d

        disp_mat[r, c] = min_disp


def multi_window(disp_mat_gpu, cost_maps_gpu, patch_size):
    rad = patch_size // 2
    multi_win[(30, 30), (30, 30)](disp_mat_gpu, cost_maps_gpu, rad)
    return disp_mat_gpu
