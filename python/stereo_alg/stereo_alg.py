import cv2 as cv
import numpy as np
import math
import stereo_alg as sa
from numba import cuda


@cuda.jit
def ssd(mat1, mat2, cost_maps, d, r):
    y, x = cuda.grid(2)
    numer = 0
    denom1 = 0
    denom2 = 0

    # 以（x, y）为中心的小窗口计算 first
    if x < mat1.shape[0] and y + d < mat1.shape[1]:
        for i in range(-r, r + 1):
            if cost_maps.shape[0] <= x + i < 0:
                continue
            for j in range(-r, r + 1):
                if cost_maps.shape[1] <= y + j + d < 0:
                    continue
                numer += (mat2[x + i, y + j] - mat1[x + i, y + j + d]) ** 2
                denom1 += mat1[x + i, y + j] ** 2
                denom2 += mat2[x + i, y + j + d] ** 2
        cost_maps[x, y] = math.sqrt(numer / (denom1 * denom2))
    cuda.syncthreads()


@cuda.jit
def compare(cost_maps, right_cost_maps):
    x, y = cuda.grid(2)

    if x + r < cost_maps.shape[0] and y + r < cost_maps.shape[1]:
        if x - r > 0 and y - r > 0:
            # 与八个方向相比较
            if cost_maps[x, y] > cost_maps[x, y - r]:
                cost_maps[x, y] = cost_maps[x, y - r]

            if cost_maps[x, y] > cost_maps[x, y + r]:
                cost_maps[x, y] = cost_maps[x, y + r]

            if cost_maps[x, y] > cost_maps[x - r, y]:
                cost_maps[x, y] = cost_maps[x - r, y]

            if cost_maps[x, y] > cost_maps[x + r, y]:
                cost_maps[x, y] = cost_maps[x + r, y]

            if cost_maps[x, y] > cost_maps[x - r, y - r]:
                cost_maps[x, y] = cost_maps[x - r, y - r]

            if cost_maps[x, y] > cost_maps[x + r, y - r]:
                cost_maps[x, y] = cost_maps[x + r, y - r]

            if cost_maps[x, y] > cost_maps[x - r, y + r]:
                cost_maps[x, y] = cost_maps[x - r, y + r]

            if cost_maps[x, y] > cost_maps[x + r, y + r]:
                cost_maps[x, y] = cost_maps[x + r, y + r]

            # right
            if right_cost_maps[x, y] > right_cost_maps[x, y - r]:
                right_cost_maps[x, y] = right_cost_maps[x, y - r]

            if right_cost_maps[x, y] > right_cost_maps[x, y + r]:
                right_cost_maps[x, y] = right_cost_maps[x, y + r]

            if right_cost_maps[x, y] > right_cost_maps[x - r, y]:
                right_cost_maps[x, y] = right_cost_maps[x - r, y]

            if right_cost_maps[x, y] > right_cost_maps[x + r, y]:
                right_cost_maps[x, y] = right_cost_maps[x + r, y]

            if right_cost_maps[x, y] > right_cost_maps[x - r, y - r]:
                right_cost_maps[x, y] = right_cost_maps[x - r, y - r]

            if right_cost_maps[x, y] > right_cost_maps[x + r, y - r]:
                right_cost_maps[x, y] = right_cost_maps[x + r, y - r]

            if right_cost_maps[x, y] > right_cost_maps[x - r, y + r]:
                right_cost_maps[x, y] = right_cost_maps[x - r, y + r]

            if right_cost_maps[x, y] > right_cost_maps[x + r, y + r]:
                right_cost_maps[x, y] = right_cost_maps[x + r, y + r]
    cuda.syncthreads()


img1 = cv.imread('trainingQ/Motorcycle/im0.png', 0).astype('float32')
img2 = cv.imread('trainingQ/Motorcycle/im1.png', 0).astype('float32')
img1_gpu = cuda.to_device(img1)
img2_gpu = cuda.to_device(img2)

rows, cols = img1.shape
N = rows * cols
maxdisp = 64
patch_size = 7
r = patch_size // 2
cost_maps = np.full([maxdisp, rows, cols], np.inf)
cost_map = np.full([rows, cols], np.inf)
cost_maps_gpu = cuda.to_device(cost_maps)
cost_map_gpu = cuda.to_device(cost_map)


for d in range(0, maxdisp):
    ssd[(30, 30), (30, 30)](img1_gpu, img2_gpu, cost_maps_gpu[d], d, r)


cost_maps = cost_maps_gpu.copy_to_host()

dispmat1 = sa.cost_to_disp(cost_maps)
right_cost_maps = sa.get_right_cost_maps(cost_maps)
right_dispmat = sa.cost_to_disp(right_cost_maps)
right_dispmat_gpu = cuda.to_device(right_dispmat)
dispmat_gpu = cuda.to_device(dispmat1)

compare[(30, 30), (30, 30)](dispmat_gpu, right_dispmat_gpu)

dispmat = dispmat_gpu.copy_to_host()
right_dispmat = right_dispmat_gpu.copy_to_host()

dispmat = sa.left_right_check(dispmat, right_dispmat, 1.0)
dispmat = sa.rgb_interpolation(img1, dispmat1, dispmat)

print(dispmat)
sa.save_pfm(dispmat, 'dispmat.pfm')

