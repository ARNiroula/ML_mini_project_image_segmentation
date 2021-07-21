from os import makedirs
import numpy as np
import matplotlib.pyplot as plt
import cv2
from KMeans import KMeans
BLUE = (255,0,0)
RED = (255,0,0)

image = cv2.imread("./Images/trash.jpg")
# plt.imshow(image)

# Change color to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image[300:,:]
pixel_vals = image.reshape((-1,3))
pixel_vals = np.float32(pixel_vals)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

k = 6

km=KMeans(n_clus=k)
km.fit(pixel_vals)
clusters=km.getClusters()
centroids=km.getCentroids()
# clusters=clusters[::-1]
centroids=np.uint8(centroids)

retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
print(labels.flatten())
print(clusters)
# centers = np.uint8(centers)
# segmented_data = centers[labels.flatten()]
segmented_data=centroids[clusters]
segmented_image = segmented_data.reshape((image.shape))
labels_reshape = labels.reshape(image.shape[0], image.shape[1])

cluster = 4

masked_image = np.copy(image)
masked_image[labels_reshape == cluster] = [BLUE]
cv2.imwrite('./Segmented Images/trash_kmeans.png', masked_image)
# plt.show()

# plt.savefig('2_next_obj.png')
