#!/usr/bin/env python
# coding: utf-8

# In[55]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle

import lazytransform as lt
from autofeat import AutoFeatClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# In[56]:

start_time = time.time()

df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    header=None)
df.columns = [
    "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
    "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
    "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
]
X = df.iloc[:, :-1]
y = (df.iloc[:, -1] == " >50K").astype(int)

seed = 42
np.random.seed(seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)


# In[57]:


print("Before transformation")
print(X_train)
print()


# In[58]:


# LazyTransformer
lazy = lt.LazyTransformer(model=None, encoders="auto", scalers="std", verbose=2)
X_train, y_train = lazy.fit_transform(X_train, y_train)
X_test = lazy.transform(X_test)
print()


# In[59]:


print("After transformation")
print(X_train)
print()


# In[60]:


model = AutoFeatClassifier(verbose=1)
X_train_with_feature_creation = model.fit_transform(X_train, y_train)
print(X_train_with_feature_creation.head(1))


# In[ ]:


X_test_with_feature_creation = model.transform(X_test)


# In[ ]:


model_1 = LogisticRegression().fit(X_train, y_train)
model_2 = LogisticRegression().fit(X_train_with_feature_creation, y_train)


# In[ ]:


print()
print(f"accuracy with no autofeat: {accuracy_score(y_test, model_1.predict(X_test))}")
print()
print(f"accuracy with autofeat: {accuracy_score(y_test, model_2.predict(X_test_with_feature_creation))}")
print()

# In[ ]:
print(f"--- {round(time.time() - start_time, 2)} seconds ---")

pkl_filename = "model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model_2, file)


