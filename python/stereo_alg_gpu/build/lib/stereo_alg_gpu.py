import cv2 as cv
import numpy as np
import struct
import math
import functools
from numba import cuda, jit
from adapt_weight import adapt_weight


@cuda.jit
def ad(mat1, mat2, cost_map, d):
    r, c = cuda.grid(2)
    if r < mat1.shape[0] and c + d < mat1.shape[1]:
        cost_map[r, c+d] = abs(mat2[r, c] - mat1[r, c+d])
    cuda.syncthreads()


@cuda.jit
def sd(mat1, mat2, cost_map, d):
    r, c = cuda.grid(2)
    if r < mat1.shape[0] and c + d < mat1.shape[1]:
        cost_map[r, c+d] = (mat2[r, c] - mat1[r, c+d])**2
    cuda.syncthreads()


@cuda.jit
def xd(mat1, mat2, cost_map, d):
    t = 12.5
    r, c = cuda.grid(2)
    if r < mat1.shape[0] and c + d < mat1.shape[1]:
        cost_map[r, c+d] = 255 / (1 + math.exp(-(abs(mat2[r, c] - mat1[r, c+d])-t)/(0.14 * t)))
    cuda.syncthreads()


@cuda.jit
def sum(cost_map, new_cost_map, r, d):
    x, y = cuda.grid(2)
    if x < cost_map.shape[0] and d <= y < cost_map.shape[1]:
        _sum = 0
        for i in range(-r, r+1):
            if cost_map.shape[0] <= x + i < 0:
                continue
            for j in range(-r, r+1):
                if cost_map.shape[1] <= y + j < 0:
                    continue
                _sum += cost_map[x+i, y+j]
        new_cost_map[x, y] = _sum
    cuda.syncthreads()


@cuda.jit
def census(mat, result, r):
    x, y = cuda.grid(2)
    if x < cost_map.shape[0] and d <= y < cost_map.shape[1]:
        _sum = 0
        for i in range(-r, r+1):
            if cost_map.shape[0] <= x + i < 0:
                continue
            for j in range(-r, r+1):
                if cost_map.shape[1] <= y + j < 0:
                    continue
                _sum += cost_map[x+i, y+j]
        new_cost_map[x, y] = _sum
    cuda.syncthreads()


def sad(mat1_gpu, mat2_gpu, patch_size, maxdisp):
    rows, cols = mat1_gpu.shape
    cost_map = np.full([rows, cols], np.inf, dtype='float32')
    cost_maps = np.full([maxdisp, rows, cols], np.inf, dtype='float32')

    cost_map_gpu = cuda.to_device(cost_map)
    cost_maps_gpu = cuda.to_device(cost_maps)

    for d in range(maxdisp):
        ad[(30, 30), (30, 30)](mat1_gpu, mat2_gpu, cost_map_gpu, d)
        sum[(30, 30), (30, 30)](cost_map_gpu, cost_maps_gpu[d], 4, d)

    return cost_maps_gpu


def ssd(mat1_gpu, mat2_gpu, patch_size, maxdisp):
    rows, cols = mat1_gpu.shape
    cost_map = np.full([rows, cols], np.inf, dtype='float32')
    cost_maps = np.full([maxdisp, rows, cols], np.inf, dtype='float32')

    cost_map_gpu = cuda.to_device(cost_map)
    cost_maps_gpu = cuda.to_device(cost_maps)

    for d in range(maxdisp):
        sd[(30, 30), (30, 30)](mat1_gpu, mat2_gpu, cost_map_gpu, d)
        sum[(30, 30), (30, 30)](cost_map_gpu, cost_maps_gpu[d], 4, d)

    return cost_maps_gpu


def sxd(mat1_gpu, mat2_gpu, patch_size, maxdisp):
    rows, cols = mat1_gpu.shape
    cost_map = np.full([rows, cols], np.inf, dtype='float32')
    cost_maps = np.full([maxdisp, rows, cols], np.inf, dtype='float32')

    cost_map_gpu = cuda.to_device(cost_map)
    cost_maps_gpu = cuda.to_device(cost_maps)

    for d in range(maxdisp):
        xd[(30, 30), (30, 30)](mat1_gpu, mat2_gpu, cost_map_gpu, d)
        sum[(30, 30), (30, 30)](cost_map_gpu, cost_maps_gpu[d], 10, d)

    return cost_maps_gpu
