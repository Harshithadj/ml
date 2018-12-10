#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn import neighbors, datasets
iris = datasets.load_iris()
X, y = iris.data, iris.target
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
result = knn.predict([[3, 5, 4, 2],])
print(iris.target_names[result])

