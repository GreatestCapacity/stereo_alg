import cv2 as cv
import numpy as np
import struct
import math
import functools
from numba import cuda, jit


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
    r, c = cuda.grid(2)
    if r < mat1.shape[0] and c + d < mat1.shape[1]:
        cost_map[r, c+d] = 255 / (1 + math.exp(-(abs(mat2[r, c] - mat1[r, c+d])-13.2)/1.8))
    cuda.syncthreads()


def rectangular_aggregation(mat, result, rds, d):
    @cuda.jit
    def move_add(i, j):
        r, c = cuda.grid(2)
        rows,cols = mat.shape
        if 0 <= r + i < rows and 0 <= c + j and c + j + d < cols:
            result[r, c + d] = result[r, c + d] + mat[r + i, c + j + d]

    for i in range(-rds, rds+1):
        for j in range(-rds, rds+1):
            move_add[(30, 30), (30, 30)](i, j)
    return result


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
        cost_maps_gpu[d] = rectangular_aggregation(cost_map_gpu, cost_maps_gpu[d], 4, d)

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
        sum[(30, 30), (30, 30)](cost_map_gpu, cost_maps_gpu[d], 4, d)

    return cost_maps_gpu
