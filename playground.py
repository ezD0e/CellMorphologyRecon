# Cell boundary recognization & pattern/contour roughness estimation
import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from skimage.draw import (line, polygon, disk,
                          circle_perimeter,
                          ellipse, ellipse_perimeter,
                          bezier_curve)
from skimage import (color,filters,measure,exposure,data,morphology,segmentation)
from skimage.measure import shannon_entropy
from skimage.filters import (try_all_threshold,threshold_li,
                            threshold_mean,threshold_minimum,
                            threshold_otsu,threshold_triangle,
                            threshold_yen)
from skimage import data, img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)
import seaborn as sns
from sympy import centroid
import pandas as pd

ini_img = skimage.io.imread('Z:/CodeCluster/SampleMicroPics/800px-Astrocyte5.jpg')

bw_img = color.rgb2gray(ini_img)

# create a mask
pre_mask = morphology.remove_small_objects(bw_img < 0.2, connectivity = 2)
mask = morphology.remove_small_holes(pre_mask, connectivity = 1)
#mask = morphology.remove_small_holes(
#    morphology.remove_small_objects(
#        bw_img < 0.5, 600),
#    600)

mask = morphology.opening(mask, morphology.disk(3))

# SLIC result
slic = segmentation.slic(bw_img, n_segments=200, start_label=1)

# maskSLIC result
m_slic = segmentation.slic(bw_img, n_segments=100, mask=mask, start_label=1)

# Display result
fig, ax_arr = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
ax1, ax2, ax3, ax4 = ax_arr.ravel()

ax1.imshow(bw_img)
ax1.set_title('Original image')

ax2.imshow(mask, cmap='gray')
ax2.set_title('Mask')

ax3.imshow(segmentation.mark_boundaries(bw_img, slic))
ax3.contour(mask, colors='red', linewidths=1)
ax3.set_title('SLIC')

ax4.imshow(segmentation.mark_boundaries(bw_img, m_slic))
ax4.contour(mask, colors='red', linewidths=1)
ax4.set_title('maskSLIC')

for ax in ax_arr.ravel():
    ax.set_axis_off()

plt.tight_layout()
plt.show()

# Morphological Snakes
# Morphological GAC
image = ini_img
gimage = inverse_gaussian_gradient(image)

# Initial level set
init_ls = np.zeros(image.shape, dtype=np.int8)
init_ls[10:-10, 10:-10] = 1
# List with intermediate results for plotting the evolution
evolution = []

def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store
callback = store_evolution_in(evolution)

# Check where the bug comes from this block
# iterations= replaced-> num_iter=
ls = morphological_geodesic_active_contour(gimage, iterations=230
                                           init_level_set=init_ls,
                                           smoothing=1, balloon=-1,
                                           threshold=0.69,
                                           iter_callback=callback)

ax2.imshow(image, cmap="gray")
ax2.set_axis_off()
ax2.contour(ls, [0.5], colors='r')
ax2.set_title("Morphological GAC segmentation", fontsize=12)

ax3.imshow(ls, cmap="gray")
ax3.set_axis_off()
contour = ax3.contour(evolution[0], [0.5], colors='g')
contour.collections[0].set_label("Iteration 0")
contour = ax3.contour(evolution[100], [0.5], colors='y')
contour.collections[0].set_label("Iteration 100")
contour = ax3.contour(evolution[-1], [0.5], colors='r')
contour.collections[0].set_label("Iteration 230")
ax3.legend(loc="upper right")
title = "Morphological GAC evolution"
ax3.set_title(title, fontsize=12)

fig.tight_layout()
plt.show()