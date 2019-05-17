import cv2 as cv
import stereo_alg as sa


def alg_sad_cpu(img1, img2, maxdisp):
    mat1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY).astype('float32')
    mat2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY).astype('float32')

    patch_size = 9

    cost_maps = sa.sad(mat1, mat2, patch_size, maxdisp)
    disp_mat = sa.cost_to_disp(cost_maps)

    return disp_mat

