import math
import cv2 as cv
import numpy as np
from numba import cuda

__gamma_c = 5.0
__gamma_p = 17.5
__T = 40.0
__width = 35.0
__height = 35.0


@cuda.jit('float32(float32[:], float32[:])', device=True)
def dist_l1(v1, v2):
    dim = v1.shape[0]
    _sum = 0.0
    for i in range(dim):
        _sum += abs(v1[i] - v2[i])
    return _sum


@cuda.jit('float32(float32[:], float32[:])', device=True)
def dist_l2(v1, v2):
    dim = v1.shape[0]
    _sum = 0.0
    for i in range(dim):
        x = v1[i] - v2[i]
        _sum += x * x
    return math.sqrt(_sum)


@cuda.jit('float32(float32[:], float32[:])', device=True)
def c(p, q):
    return dist_l2(p, q)


@cuda.jit('float32(float32, float32)', device=True)
def g(x, y):
    return math.sqrt(x * x + y * y)


@cuda.jit('float32(float32[:], float32[:], float32, float32)', device=True)
def w(p, q, x, y):
    return math.exp(-(c(p, q) / __gamma_c + g(x, y) / __gamma_p))


@cuda.jit('float32(float32[:], float32[:])', device=True)
def e(q, qd):
    ad = dist_l1(q, qd)
    return min(ad, __T)


@cuda.jit
def matching_cost(img1_lab, img2_lab, img1_bgr, img2_bgr, cost_maps, maxdisp):
    r, c = cuda.grid(2)

    if r < img1_lab.shape[0] and c < img1_lab.shape[1]:
        pd = img2_lab[r, c]

        rh = int(__height // 2)
        rw = int(__height // 2)

        w_d = 0.0
        for i in range(-rh, rh + 1):
            if img1_lab.shape[0] <= r + i < 0:
                continue

            for j in range(-rw, rw + 1):
                if img2_lab.shape[1] <= c + j < 0:
                    continue

                qd = img2_lab[r + i, c + j]
                w_d = w(pd, qd, i, j)

        for d in range(maxdisp):
            p = img1_lab[r, c + d]
            num = 0.0
            den = 0.0
            for i in range(-rh, rh + 1):
                if img1_lab.shape[0] <= r + i < 0:
                    continue

                for j in range(-rw, rw + 1):
                    if img2_lab.shape[1] <= c + j + d or c + j < 0:
                        continue

                    q = img1_lab[r + i, c + j + d]

                    q_bgr = img1_bgr[r + i, c + j + d]
                    qd_bgr = img2_bgr[r + i, c + j]

                    tmp = w(p, q, i, j) * w_d
                    num += tmp * e(q_bgr, qd_bgr)
                    den += tmp

            E = num / den
            cost_maps[d, r, c + d] = E


def adapt_weight(img1, img2, maxdisp, gamma_c=5.0, gamma_p=17.5, T=40.0, width=35.0, height=35.0):
    global __gamma_c
    global __gamma_p
    global __T
    global __width
    global __height
    __gamma_c = gamma_c
    __gamma_p = gamma_p
    __T = T
    __width = width
    __height = height

    img1_lab = cv.cvtColor(img1, cv.COLOR_BGR2Lab).astype('float32')
    img2_lab = cv.cvtColor(img2, cv.COLOR_BGR2Lab).astype('float32')
    img1_bgr = img1.astype('float32')
    img2_bgr = img2.astype('float32')

    img1_lab_gpu = cuda.to_device(img1_lab)
    img2_lab_gpu = cuda.to_device(img2_lab)
    img1_bgr_gpu = cuda.to_device(img1_bgr)
    img2_bgr_gpu = cuda.to_device(img2_bgr)

    rows, cols, _ = img1.shape
    cost_maps = np.full([maxdisp, rows, cols], np.inf)
    cost_maps_gpu = cuda.to_device(cost_maps)

    matching_cost[(30, 30), (15, 15)](img1_lab_gpu, img2_lab_gpu, img1_bgr_gpu, img2_bgr_gpu, cost_maps_gpu, maxdisp)

    return cost_maps_gpu
