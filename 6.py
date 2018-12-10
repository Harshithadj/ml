#!/usr/bin/env python
# coding: utf-8

# In[19]:


from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[20]:


dataset=load_iris()
model=GaussianNB()
model.fit(dataset.data,dataset.target)
expected=dataset.target
predicted=model.predict(dataset.data)
print("Accuracy Score is:",accuracy_score(expected,predicted))
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))


# In[ ]:




