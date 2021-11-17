#!/usr/bin/env python
# coding: utf-8

# # importing all the dependencies

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics


# # Data collection

# In[2]:


#importing the dataset
df = pd.read_csv('C:/Users/Hemant/jupyter_codes/ML Project 1/Car price prediction/car data.csv')


# In[3]:


#to check the first 5 rows of the dataset
df.head()


# In[4]:


#to check the shape of the dataset
df.shape


# In[5]:


#to check the info of the dataset
df.info()


# In[6]:


#check the statistical measure of the dataset
df.describe()


# In[7]:


#check for any missing values
df.isnull().sum()


# In[8]:


#check for the type of car in Fuel_Type column
df['Fuel_Type'].value_counts()


# In[9]:


#check for the category in Seller_Type column
df['Seller_Type'].value_counts()


# In[10]:


#check for the category in Transmission column
df['Transmission'].value_counts()


# Encoding

# In[11]:


#convet categorical data into numerical data
encoder = LabelEncoder()

objlist = df.select_dtypes(include = 'object').columns
objlist = objlist.drop('Car_Name')

for col_name in objlist:
    df[col_name] = encoder.fit_transform(df[col_name].astype(str))


# In[12]:


df.head()


# In[13]:


#check for the type of car in Fuel_Type column after encoding
df['Fuel_Type'].value_counts()


# Splitting the data and label

# In[14]:


X = df.drop(columns = ['Car_Name', 'Selling_Price'], axis = 1)
Y = df['Selling_Price']


# In[15]:


print(X.shape, Y.shape)


# # Splitting the data into training and testing data and then model evaluation

# In[16]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .1, random_state = 2)


# # model training: linear regression

# In[17]:


model1 = LinearRegression()


# In[18]:


model1.fit(x_train, y_train)


# model evaluation

# In[19]:


#prediction on training data
training_prediction = model1.predict(x_train)

#R squared error
error_score = metrics.r2_score(training_prediction, y_train)
print('R SQUARED ERROR :', error_score)

#visualize the actual and predicted price
plt.scatter(y_train, training_prediction)
plt.xlabel('ACTUAL PRICE')
plt.ylabel('PREDICTED PRICE')
plt.title('ACTUAL PRICE VS. PREDICTED PRICE')
plt.show()


# In[20]:


#prediction on testing data
testing_prediction = model1.predict(x_test)

#R squared error
error_score_test = metrics.r2_score(testing_prediction, y_test)
print('R SQUARED ERROR :', error_score_test)

#visualize the actual and predicted price
plt.scatter(y_test, testing_prediction)
plt.xlabel('ACTUAL PRICE')
plt.ylabel('PREDICTED PRICE')
plt.title('ACTUAL PRICE VS. PREDICTED PRICE')
plt.show()


# # model training : lasso regression

# In[21]:


model2 = Lasso()

model2.fit(x_train, y_train)


# model evaluation

# In[22]:


#prediction on training data
training_prediction = model2.predict(x_train)

#R squared error
error_score = metrics.r2_score(training_prediction, y_train)
print('R SQUARED ERROR :', error_score)

#visualize the actual and predicted price
plt.scatter(y_train, training_prediction)
plt.xlabel('ACTUAL PRICE')
plt.ylabel('PREDICTED PRICE')
plt.title('ACTUAL PRICE VS. PREDICTED PRICE')
plt.show()


# In[23]:


#prediction on testing data
testing_prediction = model2.predict(x_test)

#R squared error
error_score_test = metrics.r2_score(testing_prediction, y_test)
print('R SQUARED ERROR :', error_score_test)

#visualize the actual and predicted price
plt.scatter(y_test, testing_prediction)
plt.xlabel('ACTUAL PRICE')
plt.ylabel('PREDICTED PRICE')
plt.title('ACTUAL PRICE VS. PREDICTED PRICE')
plt.show()


# In[ ]:




