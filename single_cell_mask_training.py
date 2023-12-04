from matplotlib.image import BboxImage
from matplotlib.transforms import BboxBase
import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.io
from skimage import data, segmentation, feature, future, morphology, color, measure
from skimage.draw import ellipse, circle_perimeter, polygon_perimeter
from skimage.measure import find_contours, approximate_polygon, \
    subdivide_polygon
from sklearn.ensemble import RandomForestClassifier
from functools import partial

from sympy import Equivalent

import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Read image
full_img = skimage.io.imread('Z:/CodeCluster/SampleMicroPics/800px-Astrocyte5.jpg')

img = full_img[:588, :800]
bw_img = color.rgb2gray(img)

# create a mask
mask = morphology.remove_small_holes(
    morphology.remove_small_objects(
        bw_img < 0.2, connectivity = 2),
     connectivity = 1)

mask = morphology.opening(mask, morphology.disk(3))
mask_invert = np.invert(mask)
mask_pixel = mask_invert *1
mask_bound = skimage.measure.find_contours(bw_img,mask=mask)
fig, ax = plt.subplots()
ax.imshow(mask_pixel, cmap=plt.cm.gray)

# Mask region properties
labels = measure.label(mask_pixel)
props = measure.regionprops(labels, img)
properties = ['area', 'centroid','eccentricity', 'perimeter', 'intensity_mean']
# unfinished





# Build rect array of labels for training the segmentation.
training_labels = np.zeros(img.shape[:2], dtype=np.uint8)
training_labels[mask] = 1
training_labels[:165,:136] = 2
training_labels[450:588, :365] = 2
training_labels[:200,700:800] = 2
training_labels[164:210,464:506] = 2
training_labels[497:,611:] = 2
training_labels[316:400,658:747] = 2
sigma_min = 0.1
sigma_max = 0.9
features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=True, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max,
                        )
features = features_func(bw_img)
clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                             max_depth=5000, max_samples=2500,
                             oob_score=True,random_state=37)
clf = future.fit_segmenter(training_labels, features, clf)
result = future.predict_segmenter(features, clf)

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
ax[0].imshow(segmentation.mark_boundaries(bw_img, result, mode='thick'))
ax[0].contour(training_labels)
for item in mask_bound:
    ax[0].contour(item)
ax[0].set_title('Image, mask and segmentation boundaries')
ax[1].imshow(result)
ax[1].set_title('Segmentation')
fig.tight_layout()
plt.show()