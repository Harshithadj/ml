
# coding: utf-8

# In[2]:


import numpy as np
from urllib.request import urlopen
import urllib
import matplotlib.pyplot as plt # Visuals
import seaborn as sns 
import sklearn as skl
import pandas as pd


# In[3]:


Cleveland_data_URL = r'C:\Users\Computer\Desktop\ML LAB\DATA SET\heartdisease.csv'


# In[4]:


np.set_printoptions(threshold=np.nan)


# In[6]:


names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'heartdisease']


# In[14]:


heartDisease = pd.read_csv(Cleveland_data_URL,names = names)


# In[15]:


heartDisease.head()
del heartDisease['ca']
del heartDisease['slope']
del heartDisease['thal']
del heartDisease['oldpeak']


# In[16]:


from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator


# In[17]:


model = BayesianModel([('age', 'trestbps'), ('age', 'fbs'), ('sex', 'trestbps'), ('heartdisease','chol')])


# In[18]:


model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)


# In[19]:


print(model.get_cpds('age'))
print(model.get_cpds('chol'))
print(model.get_cpds('sex'))
model.get_independencies()


# In[20]:


from pgmpy.inference import VariableElimination
HeartDisease_infer = VariableElimination(model)


# In[21]:


q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 28})
print(q['heartdisease'])

