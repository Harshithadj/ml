#!/usr/bin/env python
# coding: utf-8

# In[20]:


# # DECISION TREE IMPLEMENTATION- ENTROPY AND INFO GAIN
import pandas as pd
from pandas import DataFrame
#with open('./PlayTennis.csv','r') as df_tennis:
#df_tennis = DataFrame.read_csv(r"./PlayTennis.csv")
df_tennis=pd.DataFrame.from_csv('./PlayTennis.csv')
df_tennis


# In[21]:



def entropy(probs):
 import math
 return sum( [-prob*math.log(prob, 2) for prob in probs] )


# In[22]:



def entropy_of_list(a_list):
 from collections import Counter
 cnt = Counter(x for x in a_list)
 print("No and Yes Classes:",a_list.name,cnt)
 num_instances = len(a_list)*1.0
 probs = [x / num_instances for x in cnt.values()]
 return entropy(probs) 

# Call Entropy:
total_entropy = entropy_of_list(df_tennis['PlayTennis'])
print("Entropy of given PlayTennis Data Set:",total_entropy)


# In[23]:



def information_gain(df, split_attribute_name, target_attribute_name,trace=0):
    df_split = df.groupby(split_attribute_name)
    nobs = len(df.index) * 1.0
    df_agg_ent = df_split.agg({target_attribute_name : [entropy_of_list,lambda x: len(x)/nobs] })[target_attribute_name]
    df_agg_ent.columns = ['Entropy', 'PropObservations']
    new_entropy = sum(df_agg_ent['Entropy'] *
    df_agg_ent['PropObservations'] )
    old_entropy = entropy_of_list(df[target_attribute_name])
    return old_entropy - new_entropy
    
     
    

print('Info-gain for Outlook is :'+str( information_gain(df_tennis,'Outlook', 'PlayTennis')),"\n")
print('\n Info-gain for Humidity is: ' + str( information_gain(df_tennis,'Humidity', 'PlayTennis')),"\n")
print('\n Info-gain for Wind is:' + str( information_gain(df_tennis,'Wind', 'PlayTennis')),"\n")
print('\n Info-gain for Temperature is:' + str(
information_gain(df_tennis , 'Temperature','PlayTennis')),"\n")


# In[24]:



def id3(df, target_attribute_name, attribute_names, default_class=None):
    from collections import Counter
    cnt = Counter(x for x in df[target_attribute_name])
    if len(cnt) == 1:
        return next(iter(cnt))
    elif df.empty or (not attribute_names):
        return default_class
    else:
        gainz = [information_gain(df, attr, target_attribute_name) for attr in attribute_names]
        index_of_max = gainz.index(max(gainz))
        best_attr = attribute_names[index_of_max]
        tree = {best_attr:{}}
        remaining_attribute_names = [i for i in attribute_names if i !=best_attr]
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset, target_attribute_name, remaining_attribute_names,default_class)
            tree[best_attr][attr_val] = subtree
            return tree
 
 
 
 
 
        
     

 


# In[25]:




attribute_names = list(df_tennis.columns)
print("List of Attributes:", attribute_names)
attribute_names.remove('PlayTennis')
print("Predicting Attributes:", attribute_names)
from pprint import pprint
tree = id3(df_tennis,'PlayTennis',attribute_names)
print("\n\nThe Resultant Decision Tree is :\n")
pprint(tree) 


# In[28]:


def classify(instance, tree, default=None):
    attribute = next(iter(tree))#tree.keys()[0]
    if instance[attribute] in tree[attribute].keys():
        result = tree[attribute][instance[attribute]]
        if isinstance(result, dict):
            return classify(instance, result)
        else:
            return result 
    else:
            return default


df_tennis['predicted'] = df_tennis.apply(classify, axis=1, args=(tree,'No' ) )


#	classify func allows for a default arg: when tree doesn't have answe r for a particular

#	combitation of attribute-values, we can use 'no' as the default guess


print('Accuracy is:' + str( sum(df_tennis['PlayTennis']==df_tennis['predicted'] ) / (1.0*len(df_tennis.index)) ))


df_tennis[['PlayTennis', 'predicted']]


# In[ ]:




