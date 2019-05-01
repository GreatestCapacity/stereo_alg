################################################################
# Project Name: stereo_alg_gpu                                 #
# Author: GreatestCapacity                                     #
# E-Mail: greatest_capacity@mail.com                           #
# Url: https://github.com/GreatestCapacity                     #
################################################################

+++++++++++++++++++++++++++++++++++
+          INTRODUCTION           +
+++++++++++++++++++++++++++++++++++

This project contains some stereo algorithms especially stereo
matching algorithms. The reason I write this project is that
when I wrote my graduation thesis I searched these algorithms
on Internet for comparing my algorithm with them, but not a bit
useful code found. So I can only implement these algorithms by
myself.

++++++++++++++++++++++++++++++++++
+         DATA STRUCTURE         +
++++++++++++++++++++++++++++++++++

CostMaps:

 im0 based cost_maps      im1 based cost_maps
+++++++++                    +++++++++++++
++++++++++                  ++++++++++++++
+++++++++++                +++++++++++++++   height: maxdisp
++++++++++++              ++++++++++++++++
+++++++++++++            +++++++++++++++++
   columns                    columns

CostMaps is a minpy.numpy.ndarray that contains matching cost maps
and it's shape is (maxdisp, rows, cols). Every element is a cost
map and each cost map has the same disparity value. The empty element
of CostMaps will be replaced with minpy.numpy.inf.

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

