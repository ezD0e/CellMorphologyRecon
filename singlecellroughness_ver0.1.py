# Single cell roughness test
# Suppose input image is only about one single cell
# Roughness is obtained via least squares method
import math
import numpy as np
import matplotlib.pyplot as plt
# Import required libraries
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Load the cell image
img = cv2.imread('Z:/CodeCluster/SampleMicroPics/cell membrane contour.png')
#img = cv2.imread('Z:/CodeCluster/ProcessedPics/C1-Probability maps.png')

# Convert the image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image to segment the cell
threshold = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Find the contours of the segmented cell
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract the cell perimeter contour
perimeter = contours[0]
perimeter_dim2 = perimeter.reshape(19,2)

# Discretize the cell perimeter using a polar grid
r = np.linspace(0, 1, 100)
theta = np.linspace(0, 2*np.pi, 100)
r_grid, theta_grid = np.meshgrid(r, theta)
x = r_grid * np.cos(theta_grid)
y = r_grid * np.sin(theta_grid)
points = np.stack((x, y), axis=2)
points_dim2 = points.reshape(10000,2)

# Use K-Means clustering to assign each discretized point to the nearest point on the cell perimeter
kmeans = KMeans(n_clusters=len(perimeter)).fit(perimeter_dim2)
labels = kmeans.predict(points_dim2)

# Calculate the local curvature at each discretized point
curvature = np.zeros(points.shape[:1])

for i in range(points.shape[-1]):
        # Get the index of the nearest point on the cell perimeter
        index = labels[i]

        # Calculate the local curvature using the three nearest points on the cell perimeter
        p1, p2, p3 = perimeter[index-1], perimeter[index], perimeter[index+1]
        # Define the function to calculate the curvature of a circle fitted to three points
        def calculate_curvature(p1, p2, p3):
            # Calculate the curvature using the formula:
            # curvature = |(p1 - p2) x (p3 - p2)| / |p1 - p2|^3
            curvature = np.abs(np.cross(p1 - p2, p3 - p2)) / np.linalg.norm(p1 - p2) ** 3
        curvature[i] = calculate_curvature(p1, p2, p3)

# Calculate the average curvature
avg_curvature = np.mean(curvature)

# Calculate the roughness as the standard deviation of the local curvature
roughness = np.std(curvature)

# Normalize the roughness by the average curvature
normalized_roughness = roughness / avg_curvature

# Print the calculated roughness
print(f'The roughness of the cell perimeter is: {roughness:.3f}')
print(f'The normalized roughness of the cell perimeter is: {normalized_roughness:.3f}')


