# stereo_alg
## Introduction

This is a library for stereo vision with easy implementations of some stereo algorithms I encountered with. I write this project for my bachelor's thesis and aim to keep doing research in this field.

So far, there is just a Python library. Next step, except for adding new algorithms to the Python library, I am going to implement these algorithms on CUDA. I have to confess that Python runs more slowly than many other languages, but this library uses Numpy as far as possible and apply Numba JIT to loops so that it is also pretty fast.

## Requirements

- Python3
- setuptools
- opencv-python
- Numpy
- Scipy
- Numba

## Installation

```bash
git clone https://github.com/GreatestCapacity/stereo_alg.git
cd stereo_alg/python/stereo_alg/
sudo python setup.py install
```

## Usage

```python
import cv2 as cv
import numpy as np
import stereo_alg as sa

patch_size = 9
maxdisp = 69

im0 = cv.imread('im0.png', cv.IMREAD_GRAYSCALE)
im1 = cv.imread('im1.png', cv.IMREAD_GRAYSCALE)

cost_maps = sa.sad(im0, im1, patch_size, maxdisp)
disp_mat = sa.cost_to_disp(cost_maps)

disp_mat = 255 * disp_mat / np.max(disp_mat)
cv.imshow('Disparity Map', disp_mat.astype('uint8'))
```

There are many samples in samples directory.

## Supported Algorithms and Functions

- Read and write .pfm file
- Compute Hamming distance between two int variable
- Rank and Census Transform[^1]
- Disparity computation
- Disparity computation method from a paper[^2]
- Left-right consistency check
- Disparity interpolation according to Euclidean distance in RGB space
- SXD, a general method to customize "Sum, X, Differences"
- SAD, Sum of Absolute Differences
- ZSAD, Zero-mean Sum of Absolute Differences
- SSD, Sum of Squared Diferences
- NCC, Normalized Cross Correlation

## CostMaps Data Structure

```
im0 based cost_maps      im1 based cost_maps
+++++++++                    +++++++++++++
++++++++++                  ++++++++++++++
+++++++++++                +++++++++++++++   height: maxdisp
++++++++++++              ++++++++++++++++
+++++++++++++            +++++++++++++++++
   columns                    columns

CostMaps is a numpy.ndarray that contains matching cost maps and
it's shape is (maxdisp, rows, cols). Every element is a cost map
and each cost map has the same disparity value. The empty element
of CostMaps will be replaced with numpy.inf.

disparity 0:
+++++++++++++ im0
+++++++++++++ im1
+++++++++++++ cost_map

disparity 1:
 +++++++++++++ im0
+++++++++++++  im1
 ++++++++++++  cost_map

disparity 2:
  +++++++++++++ im0
+++++++++++++   im1
  +++++++++++   cost_map

cost_maps:
+++++++++++    disparity: 2
++++++++++++   disparity: 1
+++++++++++++  disparity: 0
```

## References

[^1]:R. Zabih and J. Woodfill, “Non-parametric Local Transforms for Computing Visual Correspondence,” in *Proceedings of the Third European Conference on Computer Vision*, Berlin, Heidelberg, 1994, pp. 151–158.

[^2]:R. K. Gupta and S.-Y. Cho, “Window-based approach for fast stereo correspondence,” *IET Computer Vision*, vol. 7, no. 2, pp. 123–134, Apr. 2013.

