import numpy as np
import matplotlib.pyplot as plt
from KMeans import KMeans
import cv2
from scipy.spatial.distance import cdist


#Image Name Choosing
print('Enter the Photo Name:')
name=input()


#Elbow Method
def elbow_method(X):
    global name
    distortions=[]
    K=range(1,10)
    for k in K:
        model = KMeans(n_clus=k)
        model.fit(X)
        centroids=model.getCentroids()

        # print(centroids)
        distortions.append(sum(np.min(cdist(X, centroids,'euclidean'), axis=1)) / X.shape[0])
        print(distortions[-1])
    # print(distortions)

    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.savefig('./Segmented Images/elbow_method_'+name+'.png')
    plt.clf()

# K clusters with random initialization
def segmentation(X,k):
    global name
    km = KMeans(n_clus = k)
    km.fit(X)
    centers = km.getCentroids()
    clusters = km.getClusters()
    # print(centers)
    # print(clusters)
    segmented_image = centers[clusters]
    segmented_image = segmented_image.reshape((image.shape))


    plt.imshow((segmented_image).astype(np.uint8))
    plt.xlabel('k = ' + str(k))
    plt.tick_params(labelleft=False, labelbottom=False, labelright=False, labeltop=False)
    plt.savefig("./Segmented Images/kmeans_"+name+".png")
    plt.clf()




#   Image Segmentation


print("Now enter the number of cluster to be used")
k=int(input())


image = cv2.imread('./Images/'+name)  #Read image
# print("IMAGE:::::",image)
# print(image.reshape(-1,3))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Convert to RGB

X = image.reshape((-1,3))   #Reshape to (Npts, Ndim = 3)
X = np.float32(X)
# elbow_method(X)
segmentation(X,k)




