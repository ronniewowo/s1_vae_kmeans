"""Amended from Keras variational autoencoder template
Unsupervised classification (K-means clustering) using latent values

Input: latent.npy
Output: latent_class2.png
        latent_class2.npy
        
"""

from sklearn.cluster import KMeans
import xarray as xr
import numpy as np
import math 
import earthpy.plot as ep
import matplotlib.pyplot as plt

#Load data
latent = np.load('latent.npy')

#Define number of classs and perform k-means clusterring
n=2
kmeans = KMeans(n_clusters=n, random_state=0).fit(latent)
latent_class = kmeans.labels_

#Save the labels
np.save('latent_class2', latent_class)

# reshape for outputting raster image
INVALID=-999
x_arr_len = 162
y_arr_len = 59
latent_class_raster = np.ones((y_arr_len,x_arr_len))*INVALID
for i in range(0, x_arr_len):
  for j in range(0, y_arr_len):
    ind=i*y_arr_len+j
    latent_class_raster[j,i]=latent_class[ind]
    
ep.plot_bands(latent_class_raster,
              title='n_clusters='+str(n),
              scale=False)

plt.savefig('latent_class2.png')

