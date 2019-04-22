import cv2 as cv
import numpy as np
import stereo_alg as sa


def alg_ncc(img1, img2, patch_size, maxdisp):
    mat1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY).astype('float32')
    mat2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY).astype('float32')

    cost_maps = sa.ncc(mat1, mat2, patch_size, maxdisp)
    disp_mat = sa.cost_to_disp(cost_maps)

    return disp_mat

