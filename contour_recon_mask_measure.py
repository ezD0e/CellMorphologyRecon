import matplotlib.pyplot as plt
from matplotlib.image import BboxImage
from matplotlib.transforms import BboxBase
from matplotlib.patches import Ellipse, Circle
import numpy as np
import pandas as pd
import math
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
regions = measure.regionprops(labels, img)
properties = ['area', 'centroid','eccentricity', 'orientation', 'intensity_mean']
for props in regions:
    y0, x0 = props.centroid
    orientation = props.orientation
    x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
    y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
    x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
    y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

    ax.plot((x0, x1), (y0, y1), '-r', linewidth=1.5)
    ax.plot((x0, x2), (y0, y2), '-r', linewidth=1.5)
    ax.plot(x0, y0, '.g', markersize=10)
    # Draw a bounding box  
    minr, minc, maxr, maxc = props.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-b', linewidth=1) 

    #diameter = props.equivalent_diameter # Draw a circle with equivalent area to mask
    #ax.plot(x0,y0,'bo')

    #rr,cc = ellipse(x0,y0,props.major_axis_length/2,props.minor_axis_length/2,rotation=np.deg2rad(props.orientation)) #画椭圆
    #img[rr,cc] = 0
    elli = Ellipse(xy = (x0, y0), width = props.major_axis_length, height = props.minor_axis_length, angle = props.orientation, facecolor= 'yellow', alpha=0.8)
    ax.add_patch(elli)
    ax.plot(x0, y0, 'ro')

plt.show()






# contour highlight/measure
contours = measure.find_contours(mask)
for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    print("Number of coordinates per contour:", len(contour))
# Observe the result and the largest contour shall be list the first by default
plt.show()

bounding_boxes = [] 
first = True # only bound the largest contour
for contour in contours:
    if first:
        first = False
        Xmin = np.min(contour[:,0])
        Xmax = np.max(contour[:,0])
        Ymin = np.min(contour[:,1])
        Ymax = np.max(contour[:,1])
    
    bounding_boxes.append([Xmin, Xmax, Ymin, Ymax])

with_boxes  = np.copy(img)

for box in bounding_boxes:
    #[Xmin, Xmax, Ymin, Ymax]
    r = [box[0],box[1],box[1],box[0], box[0]]
    c = [box[3],box[3],box[2],box[2], box[3]]
    rr, cc = polygon_perimeter(r, c, with_boxes.shape)
    with_boxes[rr, cc] = 1 #set color white

plt.imshow(with_boxes, interpolation='nearest', cmap=plt.cm.gray)
plt.show()




# Bug in this step
#fig = ax.imshow(img)
#fig.update_traces(hoverinfo='skip') 



# For each label, add a filled scatter trace for its contour,
# and display the properties of the label in the hover of this trace.
for index in range(1, labels.max()):
    label_i = props[index].label
    contour = measure.find_contours(labels == label_i, 0.5)[0]
    y, x = contour.T
    hoverinfo = ''
    for prop_name in properties:
        hoverinfo += f'<b>{prop_name}: {getattr(props[index], prop_name):.2f}</b><br>'
    fig.add_trace(go.Scatter(
        x=x, y=y, name=label_i,
        mode='lines', fill='toself', showlegend=False,
        hovertemplate=hoverinfo, hoveron='points+fills'))

plotly.io.show(fig)





