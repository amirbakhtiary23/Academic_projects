import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
class k_means:
  def __init__(self,no_of_clusters=4):
    self.k=no_of_clusters
  def fit(self,X:np.array,itterations=300):
    shape=X.shape
    dim0=shape[0]
    dim1=shape[1]
    maxs=np.max(X,axis=1).reshape(dim0,1)
    mins=np.min(X,axis=1).reshape(dim0,1)
    centroids=np.random.uniform(low=mins[:,0], high=maxs[:,0], size=(self.k,dim0))

    itter=0
    labels_=None
    while itter<itterations:

      itter+=1
      distances=self.calc_distances(X,dim0,dim1,centroids)
      labels = np.argmin(distances, axis=0)
      centroids_=self.update_centroids(X,labels,centroids,mins,maxs)

      if np.array_equal(labels, labels_):
        if  np.max(np.abs(centroids_- centroids)) < 0.01:
          break
      labels_=labels
      centroids=centroids_

    self.centroids=centroids
    self.itters_=itter
    self.labels=labels


  def get_labels(self):
    try :
      return self.labels
    except :
      print ("Call k_means.fit() first")

  def get_itters(self):
    try :
      return self.itters_
    except :
      print ("Call k_means.fit() first")

  def update_centroids(self,X,labels,centroids,mins,maxs):

    for i in range(self.k):
      if np.any(labels == i):

        centroids[i, :] = np.mean(X[:, labels == i], axis=1)
      else:
        centroids[i, :]=np.random.uniform(low=mins[:,0], high=maxs[:,0], size=(X.shape[0]))

    return centroids

  def calc_distances(self,X,dim0,dim1,centroids):

    distances=np.zeros(shape=(self.k,dim1))
    for i in range(self.k):
      distances[i,:] = np.linalg.norm(X- centroids[i,:].reshape(dim0,1), axis = 0)
    return distances

if __name__=="__main__":
  dataset=pd.read_csv("data.csv")
  
  clustering=k_means(5)
  clustering.fit(np.array(dataset.transpose()),50)
  
  labels=clustering.get_labels()



  plt.figure(figsize=(8, 6))

  for cluster in range(5):
      cluster_data = np.array(dataset[labels == cluster])

      plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'handcrafted k means Cluster {cluster + 1}')

  plt.title(f" Data with {5} Clusters_k_means_a" )
  plt.xlabel("Feature 1")
  plt.ylabel("Feature 2")
  plt.legend()
  plt.savefig("k_means_a.png")
  
