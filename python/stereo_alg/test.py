import cv2 as cv
import numpy as np
import time
import stereo_alg as sa

imread_mode = cv.IMREAD_GRAYSCALE
# imread_mode = cv.IMREAD_COLOR


def main(file1_name, file2_name, maxdisp):
    # Record the time when the algorithmn start
    start_time = time.clock()

    # Read the two images
    img1 = cv.imread(file1_name, imread_mode)
    img2 = cv.imread(file2_name, imread_mode)

    patch_size = 9
    cost_maps = sa.sad(img1, img2, patch_size, maxdisp)
    disp_mat = sa.cost_to_disp(cost_maps)
    right_disp_mat = sa.cost_to_right_disp(cost_maps)
    check_mat = sa.left_right_check(disp_mat, right_disp_mat)

    # Output runtime
    end_time = time.clock()
    secs = end_time - start_time
    rows, cols = img1.shape
    print('runtime: %.2fs  (%.2fs/MP)' % (secs, secs/(rows*cols/1000000.0)))

    # normalize disp_mat and show
    disp_mat_show = 255 * disp_mat / (np.max(disp_mat) - np.min(disp_mat))
    right_disp_mat_show = 255 * right_disp_mat / (np.max(right_disp_mat) - np.min(right_disp_mat))
    cv.imshow('disp_mat', disp_mat_show.astype('uint8'))
    cv.imshow('right_disp_mat', right_disp_mat_show.astype('uint8'))
    cv.waitKey(0)


if __name__ == '__main__':
    main('../trainingQ/Motorcycle/im0.png', '../trainingQ/Motorcycle/im1.png', 69)
