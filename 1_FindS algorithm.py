#!/usr/bin/env python
# coding: utf-8

# In[19]:


# # finds algorithm using random variables
import csv
attributes = [['Sunny','Rainy'],
['Warm','Cold'],
['Normal','High'],
['Strong','Weak'],
['Warm','Cool'],
['Same','Change']]


# In[20]:


num_attributes = len(attributes)
print (" \n The most general hypothesis : ['?','?','?','?','?','?']\n")
print ("\n The most specific hypothesis : ['0','0','0','0','0','0']\n")


# In[21]:


a = []
filename = open(r'./finds.csv')
reader = csv.reader(filename)
for row in reader:
    a.append (row)


# In[22]:


print("\n The initial value of hypothesis: ")
hypothesis = ['0'] * num_attributes
print(hypothesis)


# In[25]:


# Comparing with First Training Example
for j in range(num_attributes):
    hypothesis[j] = a[0][j];
print(hypothesis)


# In[26]:


# Comparing with Remaining Training Examples of Given Data Set
print("\n Find S: Finding a Maximally Specific Hypothesis\n")
for i in range(len(a)):
    if a[i][num_attributes]=='Yes':
        for j in range(num_attributes):
            if a[i][j]!=hypothesis[j]:
                hypothesis[j]='?'
            else :
                hypothesis[j]= a[i][j]
    #print(" For Training Example No :{0} the hypothesis is ".format(i),hypothesis)
print("\n The Maximally Specific Hypothesis for a given Training Examples :\n")
print(hypothesis)


# In[ ]:





# In[ ]:




