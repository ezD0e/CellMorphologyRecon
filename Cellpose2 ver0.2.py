import cellpose
from cellpose.io import imread

from cellpose import models, io, utils, plot
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import measure

# Load the cell image
image = imread('Z:\CodeCluster\DingPic2022Dec\A1.tif')

# Initialize the Cellpose model with default parameters
model = models.Cellpose(gpu=True, model_type='cyto')

# Set the diameter of the cells to be segmented (in pixels)
model.diameter = 30

# Set the model to segment round cells
#model.set_params(model_type='cyto')

# Segment the cells in the image
masks, flows, styles, diams = model.eval(image, diameter=None, channels=[0,0],
                                         flow_threshold=0.4, do_3D=False)

# Or use
#masks, flows, styles = model.segment(image)

# Print the number of segmented cells
print(len(masks))

# Allocate label to each cell
labels = []
for i, mask in enumerate(masks):
    labels.append(i)

# Extract the cell regions from the original image using the masks
cell_regions = [image * mask[:,None] for mask in masks]

# Switch to skimage
# Identify the separate objects in the mask
label_map = measure.label(masks, background=0)

# Compute the properties of the objects
props = measure.regionprops(label_map)

# Iterate over the objects and print their bounding boxes
for prop in props:
    print("Bounding box:", prop.bbox)

# Compute the bounding box of the mask
#bbox = measure.bbox(masks)  <-error

#print("Bounding box:", bbox)

# Skimage section ends

# Compute the centroids, areas, and bounding boxes of the cells
centroids = []
areas = []
bboxes = []
# Iterate over the pixels in the mask again and compute the properties of the objects
for row in range(mask.shape[0]):
    for col in range(mask.shape[1]):
        label = label_map[row, col]
        if label > 0:
            # Initialize variables to store the sum of the row and column coordinates and the number of pixels
            # belonging to the object
            sum_row = 0
            sum_col = 0
            num_pixels = 0
            # Initialize variables to store the minimum and maximum row and column coordinates
            min_row = float('inf')
            max_row = float('-inf')
            min_col = float('inf')
            max_col = float('-inf')
            # Update the variables for this pixel
            sum_row += row
            sum_col += col
            num_pixels += 1
            min_row = min(min_row, row)
            max_row = max(max_row, row)
            min_col = min(min_col, col)
            max_col = max(max_col, col)
        # If this is the last pixel belonging to the object, compute the centroid, area, and bounding box
        elif label_map[row, col] == 0 and label_map[row, col-1] > 0:
            # Compute the centroid
            centroid_row = sum_row / num_pixels
            centroid_col = sum_col / num_pixels
            centroids.append((centroid_row, centroid_col))
            # Compute the area
            areas.append(num_pixels)
            # Compute the bounding box
            bbox = (min_row, min_col, max_row, max_col)
            bboxes.append(bbox)

# Print the properties of the objects
for centroid, area, bbox in zip(centroids, areas, bboxes):
    print("Centroid:", centroid)
    print("Area:", area)
    print("Bounding box:", bbox)

# Extract the cell regions from the original image using the masks
cell_regions = [image * mask[:,:,None] for mask in masks]

# Display the original image and the segmented cells
import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()
for region in cell_regions:
    plt.imshow(region)
plt.show()
