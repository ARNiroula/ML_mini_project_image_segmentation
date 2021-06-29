import numpy as np
import matplotlib.pyplot as plt
from KMeans import KMeans
import cv2

#   Image Segmentation

print('Enter the Photo Name:')
name=input()
print("Now enter the number of cluster to be used")
k=int(input())


image = cv2.imread('./'+name)  #Read image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Convert to RGB

X = image.reshape((-1,3))   #Reshape to (Npts, Ndim = 3)
X = np.float32(X)

# K clusters with random initialization
# k =4
km = KMeans(n_clus = k)
km.fit(X)
centers = km.getCentroids()
clusters = km.getClusters()

segmented_image = centers[clusters]

segmented_image = segmented_image.reshape((image.shape))


plt.imshow((segmented_image).astype(np.uint8))
plt.xlabel('k = ' + str(k))
plt.tick_params(labelleft=False, labelbottom=False, labelright=False, labeltop=False)
plt.savefig("2_next_kmeans"+name+".png")

km.vizualize()