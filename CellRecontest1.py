import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from skimage.draw import (line, polygon, disk,
                          circle_perimeter,
                          ellipse, ellipse_perimeter,
                          bezier_curve)
from skimage import (color,filters,measure,exposure,data,morphology)
from skimage.measure import shannon_entropy
from skimage.filters import (try_all_threshold,threshold_li,
                            threshold_mean,threshold_minimum,
                            threshold_otsu,threshold_triangle,
                            threshold_yen)
import seaborn as sns
from sympy import centroid
import pandas as pd

gray = plt.cm.Greys_r

def median_clipping(spectrogram, number_times_larger):
    """ Compute binary image from spectrogram where cells are marked as 1 if
    number_times_larger than the row AND column median, otherwise 0
    """
    row_medians = np.median(spectrogram, axis=1)
    col_medians = np.median(spectrogram, axis=0)

    # create 2-d array where each cell contains row median
    row_medians_cond = np.tile(row_medians, (spectrogram.shape[1], 1)).transpose()
    # create 2-d array where each cell contains column median
    col_medians_cond = np.tile(col_medians, (spectrogram.shape[0], 1))

    # find cells number_times_larger than row and column median
    larger_row_median = spectrogram >= row_medians_cond*number_times_larger
    larger_col_median = spectrogram >= col_medians_cond*number_times_larger

    # create binary image with cells number_times_larger row AND col median
    binary_image = np.logical_and(larger_row_median, larger_col_median)
    return binary_image





phase_im = skimage.io.imread('Z:/CodeCluster/ProcessedPics/C2-Probability maps.png')
imMed = median_clipping(phase_im,1)
imGray = morphology.binary_erosion(imMed)

# Plot the image for examination
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.imshow(imGray)
ax.set_title('Original Pic')
plt.show()

# Generate the histogram of the image. `skimage.exposure.histogram` will return
# the values of the histogram as well as the centers of the bins.
hist, bins = skimage.exposure.histogram(imGray)

# Plot the histogram values versus the bin centers.
plt.plot(bins, hist,linewidth=1)
plt.xlabel('pixel value (a.u.)')
plt.ylabel('counts')


# Generate both histograms. 
hist_norm1, bins_norm1 = skimage.exposure.histogram(imGray)

# Plot both histograms on the same set of axes. 
plt.plot(bins_norm1, hist_norm1, label='image 1', linewidth=1,color = 'green',alpha = 0.8)


# Add labels as expected. 
plt.xlabel('normalized pixel value (a.u.)')
plt.ylabel('count')
plt.legend()


# Apply the threshold. 
thresh_val = 0.35
thresh_im1 = imGray < thresh_val


# Set up the axes for plotting.  
fig, ax = plt.subplots(nrows=1, ncols=1)

# Plot the first image
ax.imshow(thresh_im1, cmap=gray)
ax.set_title('image 1')





# Make copies of each normalized phase image. 
phase_copy1 = np.copy(imGray)


# Using the segmentation masks, color the pixels with a value of 1.0 wherever a
# segmented object exists. 
phase_copy1[thresh_im1] = 1.0


# Make an RGB image of the segmentation by generating a three dimensional array.
rgb_image1 = np.dstack((phase_copy1, imGray, imGray))


# Label each individual cell. 
im_lab, num_obj = skimage.measure.label(thresh_im1, return_num=True)
                                   
# Print out how many we identified. 
print("Number of objects found: %s" %num_obj)

# Print Shannon Entropy
num_shannon = shannon_entropy(imGray)
print('Shannon Entropy in visual: %s' %num_shannon)

#read data from binary image
props = skimage.measure.regionprops_table(im_lab, 
                            properties=['label','area',
                                        'coords','centroid',
                                        'eccentricity',
                                        'extent','perimeter',
                                        'feret_diameter_max',''])

#Save result to csv file
props_table = pd.DataFrame(props)
props_table.to_csv('Z:/CodeCluster/export.csv')

plt.matshow(im_lab)

plt.show()



