import math
from numba import cuda


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


@cuda.jit('void(float32[:, :], float32[:, :], float32[:, :])')
def sub(mat1, mat2, result):
    r, c = cuda.grid(2)
    if r < mat1.shape[0] and c < mat1.shape[1]:
        result[r, c] = mat1[r, c] - mat2[r, c]


@cuda.jit('void(float32[:, :], float32[:, :], float32[:, :])')
def mul(mat1, mat2, result):
    r, c = cuda.grid(2)
    if r < mat1.shape[0] and c < mat1.shape[1]:
        result[r, c] = mat1[r, c] * mat2[r, c]


@cuda.jit('void(float32[:, :], float32[:, :], float32[:, :])')
def div(mat1, mat2, result):
    r, c = cuda.grid(2)
    if r < mat1.shape[0] and c < mat1.shape[1]:
        result[r, c] = mat1[r, c] / mat2[r, c]


@cuda.jit('void(float32[:, :], float32[:, :])')
def absolute(mat, result):
    r, c = cuda.grid(2)
    if r < mat.shape[0] and c < mat.shape[1]:
        result[r, c] = abs(mat[r, c])


@cuda.jit('void(float32[:, :], float32[:, :])')
def square(mat, result):
    r, c = cuda.grid(2)
    if r < mat.shape[0] and c < mat.shape[1]:
        result[r, c] = mat[r, c] * mat[r, c]


@cuda.jit('void(float32[:, :], float32[:, :])')
def sqrt(mat, result):
    r, c = cuda.grid(2)
    if r < mat.shape[0] and c < mat.shape[1]:
        result[r, c] = math.sqrt(mat[r, c])


@cuda.jit('void(float32[:, :], float32[:, :], int32)')
def aggregt(mat, result, radius):
    r, c = cuda.grid(2)
    if r < mat.shape[0] and c < mat.shape[1]:
        _sum = 0
        for i in range(-radius, radius+1):
            if mat.shape[0] <= r + i or r + i < 0:
                continue
            for j in range(-radius, radius+1):
                if mat.shape[1] <= c + j or c + j < 0:
                    continue
                _sum += mat[r+i, c+j]
        result[r, c] = _sum
