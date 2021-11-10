#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd


# In[27]:


data = pd.read_excel("train(1).xlsx")


# In[28]:


data.columns


# In[29]:


# data.describe()


# In[30]:


# data.info()


# In[31]:


target = data['污水厂采样点']


# In[52]:


data1 = data.iloc[:,1:]


# In[53]:


data1


# In[54]:


data = np.array(data1)
# data = data.astype(data)


# In[56]:


# data


# In[57]:


target.shape


# In[58]:


data.shape


# In[59]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
# print type(data1)
x = data
y = target
# print type(x)
# print x.shape


# In[60]:


pca = PCA(n_components=4) 


# In[61]:


reduce_X = pca.fit_transform(x)


# In[62]:


# 查看降维后的数据分布
plt.scatter(reduce_X[:,0], reduce_X[:,1],c = y)
plt.show()


# In[ ]:




