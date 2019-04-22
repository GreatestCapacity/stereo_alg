import cv2 as cv
import numpy as np
import stereo_alg as sa


def alg_census(img1, img2, patch_size, maxdisp):
    mat1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY).astype('float32')
    mat2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY).astype('float32')

    census1 = sa.census(mat1, patch_size)
    census2 = sa.census(mat2, patch_size)

    s = lambda a: a
    x = lambda a: a
    d = np.vectorize(sa.hamming)
    cost_maps = sa.sxd(census1, census2, s, x, d, maxdisp)
    disp_mat = sa.cost_to_disp(cost_maps)

    return disp_mat

