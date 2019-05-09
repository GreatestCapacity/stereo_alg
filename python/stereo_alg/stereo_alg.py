import cv2 as cv
import numpy as np
import scipy.signal
import struct
import functools
from numba import jit

# New data type definition
CostMaps = np.ndarray
Matrix = np.ndarray


def read_pfm(file_name: str) -> Matrix:
    """
    Read PFM file to Numpy Array
    :param file_name: the file's name
    :return: the data from the file
    """
    f = open(file_name, 'rb')
    magic_num = f.readline().decode('utf-8')
    cols, rows = map(int, f.readline().decode('utf-8').split())
    scalefactor = float(f.readline().decode('utf-8'))

    mat = np.zeros([rows, cols])
    for r in range(rows, 0, -1):
        for c in range(cols):
            mat[r-1, c-1] = struct.unpack('f', f.read(4))[0]

    f.close()
    return mat


def save_pfm(data: Matrix, file_name: str, scalefactor: float = 1/255.0) -> None:
    """
    Save Image Matrix to PFM file
    :param data: the data need to be saved
    :param file_name: the file's name
    :param scalefactor: scale factor
    :return: None
    """
    height = data.shape[0]  # rows
    width = data.shape[1]  # cols
    f = open(file_name, 'wb')
    f.write(b'Pf\n%d %d\n%f\n' % (width, height, -scalefactor))
    for i in range(height-1, -1, -1):
        for j in range(width):
            f.write(struct.pack('f', data[i, j]))
    f.close()


@jit
def hamming(x: int, y: int) -> int:
    """
    Hamming Distance Computation
    :param x: binary number x
    :param y: binary number y
    :return: the hamming distance between x and y
    This function will consider the two integers as two
    binary numbers and counts the different bits.
    """
    dist = 0
    val = x ^ y
    while val:
        dist = dist + 1
        val = val & (val - 1)
    return dist


@jit
def rank(mat: Matrix, patch_size: int) -> Matrix:
    """
    Rank Transform
    :param mat: the matrix need to be transformed
    :param patch_size: an odd integer, the diameter of the square patch
    :return: the matrix has been transformed
    """
    patch_radius = patch_size // 2
    rank_mat = np.zeros_like(mat, dtype=np.int)
    rows, cols = mat.shape

    for r in range(rows):
        for c in range(cols):
            for i in range(-patch_radius+1, patch_radius):
                if r + i < 0 or r + i >= rows:
                    continue
                for j in range(-patch_radius+1, patch_radius):
                    if c + j < 0 or c + j >= cols or i == j == 0:
                        continue

                    if mat[r+i, c+j] < mat[r, c]:
                        rank_mat[r, c] = rank_mat[r, c] + 1

    return rank_mat


@jit
def census(mat: Matrix, patch_size: int) -> Matrix:
    """
    Census Transform
    :param mat: the matrix need to be transformed
    :param patch_size: an odd integer, the diameter of the square patch
    :return: the matrix has been transformed
    """
    patch_radius = patch_size // 2
    census_mat = np.zeros_like(mat, dtype=np.int)
    rows, cols = mat.shape

    for r in range(rows):
        for c in range(cols):
            for i in range(-patch_radius+1, patch_radius):
                if r + i < 0 or r + i >= rows:
                    continue
                for j in range(-patch_radius+1, patch_radius):
                    if c + j < 0 or c + j >= cols or i == j == 0:
                        continue

                    if mat[r+i, c+j] >= mat[r, c]:
                        census_mat[r, c] = census_mat[r, c] + 1

                    census_mat[r, c] = census_mat[r, c] << 1

    return census_mat


def get_right_cost_maps(cost_maps: CostMaps) -> CostMaps:
    """
    Return Right Cost Maps from Left Cost Maps
    :param cost_maps: left cost maps
    :return: right cost maps
    All the stereo matching functions return left cost maps.
    """
    maxdisp, rows, cols = cost_maps.shape
    for d in range(1, maxdisp):
        inf_mat = np.full([rows, d], np.inf)
        cost_maps[d] = np.append(cost_maps[d, :, d:], inf_mat, 1)
    return cost_maps


def cost_to_disp(cost_maps: CostMaps) -> Matrix:
    """
    Disparity Map Computation
    :param cost_maps: cost maps
    :return: disparity map
    For more information about CostMaps, see documentation.
    """
    return np.argmin(cost_maps, 0)


def cost_to_disp_with_penalty(mat: Matrix, cost_maps: CostMaps, T: float) -> Matrix:
    """
    Disparity Computation From R Gupta and S-Y Cho
    :param mat: gray image
    :param cost_maps: cost maps
    :param T: a constant from the paper
    :return: disparity map
    For more information about this function, see the journal paper below
    Gupta R K, Cho S-Y. Window-based approach for fast stereo correspondence[J].
    IET Computer Vision, 2013, 7(2): 123–134.
    """
    def one_side_compute(_mat, _cost_maps, is_left):
        maxdisp, rows, cols = _cost_maps.shape
        disp_mat = np.zeros([rows, cols])

        if is_left:
            c_range = range(1, cols)
            d0 = np.argmin(_cost_maps[:, :, 0], 0)
            disp_mat[:, 0] = d0
        else:
            c_range = range(cols-2, -1, -1)
            d0 = np.argmin(_cost_maps[:, :, cols - 1], 0)
            disp_mat[:, cols-1] = d0

        for c in c_range:
            for d in range(maxdisp):
                if is_left:
                    _c = c - 1
                else:
                    _c = c + 1
                penalty = T * np.absolute(d - d0) * (1 - np.absolute(_mat[:, c] - _mat[:, _c]) / 255)
                _cost_maps[d, :, c] = _cost_maps[d, :, c] + penalty

            d0 = np.argmin(_cost_maps[:, :, c], 0)
            disp_mat[:, c] = d0

        return disp_mat

    d_cl = one_side_compute(mat, cost_maps, True)
    d_cr = one_side_compute(mat, cost_maps, False)
    return np.min(np.array([d_cl, d_cr]), 0)


@jit
def left_right_check(left_disp_mat: Matrix, right_disp_mat: Matrix, thresh: float = 1.0) -> Matrix:
    """
    Left-Right Consistency Check
    :param left_disp_mat: disparity map based on the left image
    :param right_disp_mat: disparity map based on the right image
    :param thresh: the threshold to check if a pixel is correct matched
    :return: a boolean matrix that True strands for correct and False stands for incorrect
    """
    rows, cols = left_disp_mat.shape
    check_mat = np.full_like(left_disp_mat, False)
    for r in range(rows):
        for c in range(cols):
            d = int(left_disp_mat[r, c])
            if c - d < 0:
                continue
            if abs(d - right_disp_mat[r, c - d]) <= thresh:
                check_mat[r, c] = True

    return check_mat


@jit
def subpixel_enhance(disp_mat: Matrix, cost_maps: CostMaps) -> Matrix:
    """
    Subpixel Enhancement
    :param disp_mat: disparity map
    :param cost_maps: cost maps
    :return: disparity map
    """
    disp_mat_float = disp_mat.astype('float32')
    maxdisp, rows, cols = cost_maps.shape
    for r in range(rows):
        for c in range(cols):
            d = disp_mat[r, c]
            if d - 1 < 0 or d + 1 >= min(maxdisp, c+1):
                continue
            cost = cost_maps[d, r, c]
            cost_f = cost_maps[d+1, r, c]
            cost_b = cost_maps[d-1, r, c]
            disp_mat_float[r, c] = d - (cost_f - cost_b) / (2 * (cost_f - 2*cost + cost_b))
    return disp_mat_float


@jit
def rgb_interpolation(img: Matrix, disp_mat: Matrix, check_mat: Matrix) -> Matrix:
    """
    Disparity Interpolation using RGB distance
    :param img: RGB image
    :param disp_mat: disparity map
    :param check_mat: returned by left_right_check
    :return: disparity map interpolated
    """
    rows, cols = disp_mat.shape

    for r in range(0, rows):
        for c in range(0, cols):
            if check_mat[r, c]:
                continue

            min_dist = np.inf
            d = 0.0
            for i in range(-2, 3):
                if r + i < 0 or r + i >= rows:
                    continue
                for j in range(-2, 3):
                    if c + j < 0 or c + j >= cols or not check_mat[r+i, c+j]:
                        continue

                    dist = np.linalg.norm(img[r, c] - img[r+i, c+j])
                    if dist < min_dist:
                        min_dist = dist
                        d = disp_mat[r+i, c+j]
            disp_mat[r, c] = d
    return disp_mat


@jit
def left_interpolation(disp_mat: Matrix, check_mat: Matrix) -> Matrix:
    """
    Disparity Interpolation using RGB distance
    :param img: RGB image
    :param disp_mat: disparity map
    :param check_mat: returned by left_right_check
    :return: disparity map interpolated
    """
    rows, cols = disp_mat.shape

    for r in range(0, rows):
        for c in range(1, cols):
            if not check_mat[r, c] and disp_mat[r, c-1]:
                disp_mat[r, c] = disp_mat[r, c-1]
    return disp_mat


def sxd(mat1: Matrix, mat2: Matrix, s, x, d, maxdisp: int) -> CostMaps:
    rows = mat1.shape[0]
    cols = mat1.shape[1]
    cost_maps = np.full([maxdisp, rows, cols], np.inf)
    cost_map = s(x(d(mat2, mat1)))
    cost_maps[0] = cost_map
    for disp in range(1, maxdisp):
        cost_map = s(x(d(mat2[:, :-disp, ...], mat1[:, disp:, ...])))
        cost_maps[disp, :, disp:] = cost_map

    return cost_maps


def sad(mat1: Matrix, mat2: Matrix, patch_size: int, maxdisp: int) -> CostMaps:
    """
    Sum of Absolute Differences
    :param mat1: the left gray image
    :param mat2: the right gray image
    :param patch_size: an odd integer, the diameter of the square patch
    :param maxdisp: the max disparity
    :return: cost maps
    """
    mat1 = mat1.astype('float32')
    mat2 = mat2.astype('float32')

    kernel = np.ones((patch_size, patch_size))
    s = functools.partial(scipy.signal.convolve2d, in2=kernel, mode='same')
    d = lambda a, b: a - b
    return sxd(mat1, mat2, s, np.absolute, d, maxdisp)


def zsad(mat1: Matrix, mat2: Matrix, patch_size: int, maxdisp: int) -> CostMaps:
    """
    Zero-mean Sum of Absolute Differences
    :param mat1: the left gray image
    :param mat2: the right gray image
    :param patch_size: an odd integer, the diameter of the square patch
    :param maxdisp: the max disparity
    :return: cost maps
    """
    mat1 = mat1.astype('float32')
    mat2 = mat2.astype('float32')

    @jit
    def zero_mean(a, b):
        patch_radius = patch_size // 2
        rows, cols = a.shape
        cost_map = np.zeros_like(a)

        for r in range(rows):
            for c in range(cols):
                r_low = r - patch_radius
                r_high = r + patch_radius + 1
                c_low = c - patch_radius
                c_high = c + patch_radius + 1
                if r - patch_radius < 0:
                    r_low = 0
                elif r + patch_radius >= rows:
                    r_high = rows - 1
                if c - patch_radius < 0:
                    c_low = 0
                elif c + patch_radius >= cols:
                    c_high = cols - 1

                patch1 = a[r_low:r_high, c_low:c_high]
                patch2 = b[r_low:r_high, c_low:c_high]

                patch1 = patch1 - np.mean(patch1)
                patch2 = patch2 - np.mean(patch2)

                y = 255 / (1 + np.exp(-(np.absolute(patch1 - patch2)-12)/1.6))
                cost_map[r, c] = np.sum(y)
        return cost_map

    s = lambda a: a
    x = lambda a: a
    return sxd(mat1, mat2, s, x, zero_mean, maxdisp)


def ssd(mat1: Matrix, mat2: Matrix, patch_size: int, maxdisp: int) -> CostMaps:
    """
    Sum of Absolute Differences
    :param mat1: the left gray image
    :param mat2: the right gray image
    :param patch_size: an odd integer, the diameter of the square patch
    :param maxdisp: the max disparity
    :return: cost maps
    """
    mat1 = mat1.astype('float32')
    mat2 = mat2.astype('float32')

    kernel = np.ones((patch_size, patch_size))
    s = functools.partial(scipy.signal.convolve2d, in2=kernel, mode='same')
    d = lambda a, b: a - b
    return sxd(mat1, mat2, s, np.square, d, maxdisp)


@jit
def ncc(mat1: Matrix, mat2: Matrix, patch_size: int, maxdisp: int) -> CostMaps:
    """
    Normalized Cross Correlation
    :param mat1: the left gray image
    :param mat2: the right gray image
    :param patch_size: an odd integer, the diameter of the square patch
    :param maxdisp: the max disparity
    :return: cost maps
    """
    mat1 = mat1.astype('float32')
    mat2 = mat2.astype('float32')

    kernel = np.ones((patch_size, patch_size))

    denominator1 = scipy.signal.convolve2d(np.square(mat1), kernel, mode='same')
    denominator1 = np.sqrt(denominator1)

    denominator2 = scipy.signal.convolve2d(np.square(mat2), kernel, mode='same')
    denominator2 = np.sqrt(denominator2)

    rows, cols = mat1.shape
    cost_maps = np.full([maxdisp, rows, cols], np.inf)
    numerator = scipy.signal.convolve2d(mat2 * mat1, kernel, mode='same')
    denominator = denominator2 * denominator1
    cost_map = -(numerator / denominator)
    cost_maps[0] = cost_map
    for d in range(1, maxdisp):
        numerator = scipy.signal.convolve2d(mat2[..., :-d] * mat1[..., d:], kernel, mode='same')
        denominator = denominator2[..., :-d] * denominator1[..., d:]
        cost_map = -(numerator / denominator)
        cost_maps[d, :, d:] = cost_map

    return cost_maps
