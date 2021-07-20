from KMeans import KMeans
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


#=-=-=-=-=-=-=-=-=-=-
#   Generate data
#=-=-=-=-=-=-=-=-=-=-
classA = -2 * np.random.rand(80,2).T
classB = 1 + 2 * np.random.rand(40,2).T
classC = 3 + 2 * np.random.rand(40,2).T
# print(classA,classB,classC)
X = np.concatenate((classA, classB), axis=1)
X = np.concatenate((X, classC), axis=1).T

o = np.array([[-2, 2],[1, 3.5]]) #initialize centroids

#=-=-=-=-=-=-=-=-=-=-
#   Create and train
#   the model.
#=-=-=-=-=-=-=-=-=-=-

# Three clusters with predefined intial centroid positions
# km = KMeans(n_clus = 2)
# km.fit(X, init_state = o)

# # Two clusters with random initialization
# #km = KMeans(n_clus = 2)
# #km.fit(X)

# #=-=-=-=-=-=-=-=-=-=-
# #   Vizualize final
# #   result and predict
# #   clusters.
# #=-=-=-=-=-=-=-=-=-=-
# point_test = np.array([[2,2], [-2,-2]])
# print(km.predict(point_test))
# km.vizualize('testing')
distortions=[]


K=range(1,10)
for k in K:
    model = KMeans(n_clus=k)
    model.fit(X)
    centriods=model.getCentroids()
    print(centriods)
    distortions.append(sum(np.min(cdist(X, centriods,'euclidean'), axis=1)) / X.shape[0])
print(distortions)

plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.savefig('./Segmented Images/elbow_method_.png')
