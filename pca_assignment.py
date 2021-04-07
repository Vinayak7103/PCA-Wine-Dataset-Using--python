#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


# In[2]:


wine_df = pd.read_csv("C:/Users/vinay/Downloads/wine.csv")
wine_df.head()


# In[3]:


# Considering only numerical data 
wine_df.drop(['Type'],axis=1,inplace=True)
wine_df.describe()


# In[4]:


# Normalizing the numerical data
wine_normal=scale(wine_df)
wine_normal=pd.DataFrame(wine_normal) ##Converting from float to Dataframe format 


# In[5]:


pca=PCA(n_components=13)
pca_values=pca.fit_transform(wine_normal)


# In[6]:


var=pca.explained_variance_ratio_
var


# In[7]:


pca.components_[0]
pca.components_


# In[8]:


var1=np.cumsum(np.round(var,decimals=4)*100)
var1


# In[9]:


plt.plot(var1,color='red')


# In[10]:


pca_values


# In[11]:


### Hierarchial Clustering######
new_def=pd.DataFrame(pca_values[:,0:3])
new_def.head()


# In[12]:


from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch ##for creating dendrogram


# In[13]:


type(new_def)


# In[14]:


z=linkage(new_def,method="complete",metric="euclidean")


# plt.figure(figsize=(15, 5))
# plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('Index')
# plt.ylabel('Distance')
# sch.dendrogram(z,
#     leaf_rotation=0.,  # rotates the x axis labels
#     leaf_font_size=8.,)  # font size for the x axis labels
# plt.show()

# ## Now applying AgglomerativeClustering choosing 5 as clusters from the dendrogram

# In[15]:


from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=5,linkage='complete',affinity = "euclidean").fit(new_def) 


# In[16]:


h_complete.labels_


# In[17]:


cluster_labels=pd.Series(h_complete.labels_)
wine_df['clust']=cluster_labels # creating a  new column and assigning it to new column 
wine_df= pd.concat([cluster_labels,wine_df],axis=1)


# In[18]:


# getting aggregate mean of each cluster
wine_df.groupby(wine_df.clust).mean()


# # KMean-Clustering

# In[19]:


new_df = pd.DataFrame(pca_values[:,0:3])


# In[20]:


###### screw plot or elbow curve ##
k = list(range(2,15))
k


# In[21]:


# variable for storing total within sum of squares for each kme
TWSS = []


# In[22]:


for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(new_def)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(new_def.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,new_def.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# In[23]:


# Scree plot 
plt.plot(k,TWSS, 'ro-')
plt.xlabel("No_of_Clusters")
plt.ylabel("total_within_SS")
plt.xticks(k)


# In[24]:


# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=3) 
model.fit(new_def)


# In[25]:


model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 

wine_df['clust']=md # creating a  new column and assigning it to new column 


# In[26]:


wine_df


# In[27]:


new_def.head()


# In[28]:


wine_df.groupby(wine_df.clust).mean()


# In[ ]:




