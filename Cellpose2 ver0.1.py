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
label_map = measure.label(mask, background=0)

# Compute the properties of the objects
props = measure.regionprops(label_map)

# Iterate over the objects and print their bounding boxes
for prop in props:
    print("Bounding box:", prop.bbox)

# Compute the bounding box of the mask
bbox = measure.bbox(mask)

print("Bounding box:", bbox)
# Skimage section ends

# Compute the centroids, areas, and bounding boxes of the cells
centroids = []
areas = []
bboxes = []
for region in cell_regions:
    # Compute the centroid
    centroid = region.mean(axis=0).mean(axis=0)
    centroids.append(centroid)
    
    # Compute the area
    area = (region > 0).sum()
    areas.append(area)
    
    # Compute the bounding box
    rows, cols = (region > 0).nonzero()
    bbox = (min(cols), max(cols), min(rows), max(rows))
bboxes.append(bbox)


# Extract the cell regions from the original image using the masks
cell_regions = [image * mask[:,:,None] for mask in masks]

# Display the original image and the segmented cells
import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()
for region in cell_regions:
    plt.imshow(region)
plt.show()
