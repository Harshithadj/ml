#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
#12,20,28,18,19,29,23,33,45,51,52,51,45,53,55
df = pd.DataFrame({ 'x' : [1,1,2,1,1,0,0,0],
                  'y':    [1,2,1,1,1,1,0,0]})
y=[2,0,2,0,1,0,1,0]
KMeans = KMeans(n_clusters =3)
KMeans.fit(df)
labels = KMeans.predict(df)
labels

from sklearn.mixture import GaussianMixture
GMM = GaussianMixture(n_components=3)
GMM.fit(df)
gmm_clusters = GMM.predict(df)
gmm_clusters

plt.figure(figsize=(20,7))

plt.subplot(1, 3, 1)
plt.scatter(df['x'],df['y'], c=df['y'] , s=40)
plt.title('Actual Classes')

plt.subplot(1, 3, 2)
plt.scatter(df['x'],df['y'], c=labels, s=40)
plt.title('KMeans Clusters')

plt.subplot(1, 3, 3)
plt.scatter(df['x'],df['y'], c=gmm_clusters, s=40)
plt.title('EM Clusters')
accuracy_score(y, labels)


# In[5]:



accuracy_score(y, gmm_clusters)


# In[ ]:




