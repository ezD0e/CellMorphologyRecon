from cellpose import models, io, utils, plot
from cellpose.io import imread
import numpy as np
import matplotlib.pyplot as plt

# model_type='cyto' or model_type='nuclei'
model = models.Cellpose(gpu=True, model_type='cyto')

files = ['R:\OneDrive\CodeCluster\DingPic2022Dec\A1.tif']
imgs = [imread(f) for f in files]
masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=[0,3],
                                         flow_threshold=0.4, cellprobe_threshold=0.0,
                                         do_3D=False)
                                         # Playing around with flow_threshold / cellprobe_threshold to adjust the maximum allowed error
                                         # Larger threshold can return more ROIs

# Set output
io.masks_flows_to_seg(imgs, masks, flows, diams, file_name, channels=[0,3])

# Save .npy output
# Create dat as cache file
dat = np.load('_seg.npy', allow_pickle=True).item()

# plot image with masks overlaid
mask_RGB = plot.mask_overlay(dat['img'], dat['masks'],
                        colors=np.array(dat['colors']))

# plot image with outlines overlaid in red
outlines = utils.outlines_list(dat['masks'])
plt.imshow(dat['img'])
for o in outlines:
    plt.plot(o[:,0], o[:,1], color='r')