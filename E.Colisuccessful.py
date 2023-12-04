import math
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go
import skimage.io
from skimage.draw import (line, polygon, disk,
                          circle_perimeter,
                          ellipse, ellipse_perimeter,
                          bezier_curve)
from skimage import (color,measure,exposure)
import seaborn as sns
from sympy import centroid
import pandas as pd


phase_im = skimage.io.imread('Z:/CodeCluster/SampleMicroPics/E.Coli1.jpg')
imGray = color.rgb2gray(phase_im)
# Set the colormap for data display. 
gray = plt.cm.Greys_r

# Generate the histogram of the image. `skimage.exposure.histogram` will return
# the values of the histogram as well as the centers of the bins.
#hist, bins = skimage.exposure.histogram(imGray)

# Plot the histogram values versus the bin centers.
#plt.plot(bins, hist,linewidth=1)
#plt.xlabel('pixel value (a.u.)')
#plt.ylabel('counts')

# Threshold the image showing pixels 
#thresh_val = 0.5
#thresh_im = imGray < thresh_val

# Plot the image.
#plt.imshow(thresh_im, cmap=gray)

#*******Second Image********
#
# Load another phase contrast image.
phase_im2 = skimage.io.imread('Z:/CodeCluster/SampleMicroPics/E.Coli2.jpg')
imGray2 = color.rgb2gray(phase_im2)

# Apply the threshold value. 
#thresh_im2 = imGray2 < thresh_val

# Show the image.
#plt.imshow(thresh_im2, cmap=gray)

# Generate the histograms for each image.
#hist_im1, bins_im1 = skimage.exposure.histogram(imGray)
#hist_im2, bins_im2 = skimage.exposure.histogram(imGray2)

# Each histogram over eachother. 
#plt.plot(bins_im1, hist_im1, label='image 1', linewidth=1,color = 'green',alpha = 0.9)
#plt.plot(bins_im2, hist_im2, label='image 2', linewidth=1,color = 'blue',alpha = 0.1)
#plt.xlabel('pixel value (a.u.)')
#plt.ylabel('count')
#plt.legend()

def normalize_im(im):
    """
    Normalizes a given image such that the values range between 0 and 1.     
    
    Parameters
    ---------- 
    im : 2d-array
        Image to be normalized.
        
    Returns
    -------
    im_norm: 2d-array
        Normalized image ranging from 0.0 to 1.0. Note that this is now
        a floating point image and not an unsigned integer array. 
    """
    im_norm = (im - im.min()) / (im.max() - im.min())
    return im_norm

# Normalize both images.
phase_norm1 = normalize_im(imGray)
phase_norm2 = normalize_im(imGray2)

# Generate both histograms. 
hist_norm1, bins_norm1 = skimage.exposure.histogram(phase_norm1)
hist_norm2, bins_norm2 = skimage.exposure.histogram(phase_norm2)

# Plot both histograms on the same set of axes. 
plt.plot(bins_norm1, hist_norm1, label='image 1', linewidth=1,color = 'green',alpha = 0.8)
plt.plot(bins_norm2, hist_norm2, label='image 2', linewidth=1,color = 'blue',alpha = 0.5)

# Add labels as expected. 
plt.xlabel('normalized pixel value (a.u.)')
plt.ylabel('count')
plt.legend()



# Apply ANN to find suitable threshold.
#
#
#

# Apply the threshold. 
thresh_val = 0.35
thresh_im1 = phase_norm1 < thresh_val
thresh_im2 = phase_norm2 < thresh_val

# Set up the axes for plotting.  
fig, ax = plt.subplots(nrows=1, ncols=2)
# This generates a single row of images with two columns and assigns them to 
# a variable `ax`.

# Plot the first image
ax[0].imshow(thresh_im1, cmap=gray)
ax[0].set_title('image 1')

# Plot the second image.
ax[1].imshow(thresh_im2, cmap=gray)
ax[1].set_title('image 2')





# Make copies of each normalized phase image. 
phase_copy1 = np.copy(phase_norm1)
phase_copy2 = np.copy(phase_norm2)

# Using the segmentation masks, color the pixels with a value of 1.0 wherever a
# segmented object exists. 
phase_copy1[thresh_im1] = 1.0
phase_copy2[thresh_im2] = 1.0

# Make an RGB image of the segmentation by generating a three dimensional array.
rgb_image1 = np.dstack((phase_copy1, phase_norm1, phase_norm1))
rgb_image2 = np.dstack((phase_copy2, phase_norm2, phase_norm2))

# Show both images again using a subplot. Since these are RGB, we won't need to 
# use a colormap.
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].imshow(rgb_image1)
ax[0].set_title('image 1')
ax[1].imshow(rgb_image2)
ax[1].set_title('image 2')


#Choose one from image1 or image2

# Label each individual cell. 
im_lab, num_obj = skimage.measure.label(thresh_im2, return_num=True)
                                   
# Print out how many we identified. By eye, we expect around 25.
print("Number of objects found: %s" %num_obj)

#read data from binary image
props = skimage.measure.regionprops_table(im_lab, 
                            properties=['label','area',
                                        'extent','solidity',
                                        'coords','centroid',
                                        'eccentricity',
                                        'extent','perimeter',
                                        'feret_diameter_max',
                                        'major_axis_length','minor_axis_length',
                                        'orientation','eccentricity',
                                        'euler_number'])
#Save result to csv file
props_table = pd.DataFrame(props)
props_table.to_csv('Z:/CodeCluster/export.csv')

plt.matshow(im_lab)

plt.show()

markers = measure.label(im_lab,connectivity=1)
props_list = measure.regionprops(markers, im_lab)
properties = ['area','bbox','extent','solidity']
                
for index in range(1, markers.max()):
    label_i = props_list[index].label
    contour = measure.find_contours(markers == label_i, 0.5)[0]
    y, x = contour.T
    hoverinfo = ''
    for prop_name in properties:
        hoverinfo += f'<b>{prop_name}: {getattr(props_list[index], prop_name):.2f}</b><br>'
    fig.add_trace(go.Scatter(
        x=x, y=y, name=label_i,
        mode='lines', fill='toself', showlegend=False,
        hovertemplate=hoverinfo, hoveron='points+fills'))

plotly.io.show(fig)


