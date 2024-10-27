import time
import pandas as pd
import numpy as np
import mpi4py.MPI as mpi
from matplotlib import pyplot as plt
comm=mpi.COMM_WORLD
my_rank=comm.Get_rank()
size=comm.Get_size()
class k_means:
  def __init__(self,no_of_clusters=4,rank=0):
    self.k=no_of_clusters
    self.rank=rank
  def fit(self,X:np.array,itterations=300):
    shape=X.shape
    dim0=shape[0]
    dim1=shape[1]
    maxs=np.max(X,axis=1).reshape(dim0,1)
    mins=np.min(X,axis=1).reshape(dim0,1)
    if self.rank==0:
      
      for i in range(1,size):
        data = comm.irecv(source=i, tag=i+i)
        data=data.wait()
        data2=comm.irecv(source=i, tag=i*i)
        data2=data2.wait()
        maxs= np.concatenate([maxs, np.array(data)],axis=1)
        mins=np.concatenate([mins, np.array(data2)],axis=1)
    else :
      req=comm.isend(list(maxs), dest=0, tag=self.rank+self.rank)
      req.wait()
      req2=comm.isend(list(mins), dest=0, tag=self.rank*self.rank)
      req2.wait()

    if self.rank==0:
      maxs=np.max(maxs,axis=1).reshape(dim0,1)
      mins=np.min(mins,axis=1).reshape(dim0,1)


    if self.rank==0 : 
      centroids=np.random.uniform(low=mins[:,0], high=maxs[:,0], size=(self.k,dim0))
      centroids=centroids.flatten()
    else : 
      centroids=None
    centroids=comm.bcast(centroids, root=0)
    centroids=centroids.reshape(self.k,dim0)
    
    itter=0
    labels_=None
    if self.rank==0:
      
      while itter<itterations:
        temp=[]
        counts=np.zeros((self.k,1))
        itter+=1
        distances=self.calc_distances(X,dim0,dim1,centroids)
        labels = np.argmin(distances, axis=0)
        centroids_=self.update_centroids(X,labels,centroids,mins,maxs)
        label_counts = np.bincount(labels,minlength=self.k)
        temp.append([centroids_,label_counts])
        for i in range(1,size):
          data = comm.irecv(source=i, tag=i*i)
          data=data.wait()
          temp.append(data)
        

        for i in range(len(temp)):
          tmp=temp[i][1].reshape(self.k,1)

          temp[i][0]=np.multiply(tmp,temp[i][0])
          counts=np.add(counts,tmp)
        to_update=np.zeros((self.k,dim0))
        for i in range(size):
          to_update=np.add(to_update,temp[i][0])
        centroids=self.update_centroids_for_nan(np.divide(to_update,counts),maxs,mins,dim0)
        centroids=centroids.flatten()
        centroids=comm.bcast(centroids, root=0)
        centroids=centroids.reshape(self.k,dim0)
      distances=self.calc_distances(X,dim0,dim1,centroids)
      self.labels = np.argmin(distances, axis=0)
  
    else :
      while itter<itterations:
        itter+=1
        distances=self.calc_distances(X,dim0,dim1,centroids)
        labels = np.argmin(distances, axis=0)
        centroids=self.update_centroids(X,labels,centroids,mins,maxs)
        label_counts = np.bincount(labels,minlength=self.k)
        data_to_send=list([centroids,label_counts])
        req=comm.isend(data_to_send, dest=0, tag=self.rank*self.rank)
        req.wait()
        centroids=None
        centroids=comm.bcast(centroids, root=0)
        centroids=centroids.reshape(self.k,dim0)
      distances=self.calc_distances(X,dim0,dim1,centroids)
      self.labels = np.argmin(distances, axis=0)

    


  def get_labels(self):
    try :
      return self.labels
    except :
      print ("Call k_means.fit() first")

  def update_centroids_for_nan(self,centroids,mins,maxs,dim0):

    for i in range(self.k):
      if np.any(np.isnan(centroids[i, :])):

        centroids[i, :]=np.random.uniform(low=mins[:,0], high=maxs[:,0], size=(dim0))
      else:
        pass
        #centroids[i, :]=np.random.uniform(low=mins[:,0], high=maxs[:,0], size=(dim0))

    return centroids
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




dataset=pd.read_csv(f"data{my_rank}.csv")
clustering=k_means(5,my_rank)
clustering.fit(np.array(dataset.transpose()),1)

if my_rank==0:
    t0=time.time()
print (f"on {my_rank}")



clustering=k_means(5,my_rank)
clustering.fit(np.array(dataset.transpose()),100)

labels=clustering.get_labels()



plt.figure(figsize=(8, 6))

for cluster in range(5):
    cluster_data = np.array(dataset[labels == cluster])

    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'handcrafted k means Cluster {cluster + 1} count : {cluster_data.shape[0]}')

plt.title(f" Data with {5} Clusters_k_means_b on node {my_rank}" )
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.savefig(f"k_means_b on node {my_rank}.png")

if my_rank==0: print (f"took {(time.time()-t0)} s")
