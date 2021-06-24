#!/usr/bin/env python
# coding: utf-8

# # Lab: Titanic Survival Exploration with Decision Trees

# ## Getting Started
# In the introductory project, you studied the Titanic survival data, and you were able to make predictions about passenger survival. In that project, you built a decision tree by hand, that at each stage, picked the features that were most correlated with survival. Lucky for us, this is exactly how decision trees work! In this lab, we'll do this much quicker by implementing a decision tree in sklearn.
# 
# We'll start by loading the dataset and displaying some of its rows.

# Recall that these are the various features present for each passenger on the ship:
# - **Pclass**: Socio-economic class (1 = Upper class; 2 = Middle class; 3 = Lower class)
# - **Name**: Name of passenger
# - **Sex**: Sex of the passenger
# - **Age**: Age of the passenger (Some entries contain `NaN`)
# - **SibSp**: Number of siblings and spouses of the passenger aboard
# - **Parch**: Number of parents and children of the passenger aboard
# - **Ticket**: Ticket number of the passenger
# - **Fare**: Fare paid by the passenger
# - **Cabin** Cabin number of the passenger (Some entries contain `NaN`)
# - **Embarked**: Port of embarkation of the passenger (C = Cherbourg; Q = Queenstown; S = Southampton)  
# - (Target variable) **Survived**: Outcome of survival (0 = No; 1 = Yes)  
# 
# 

# In[53]:


# Import libraries necessary for this project
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Pretty display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# Allows the use of display() for DataFrames
from IPython.display import display


# In[54]:


# Load the dataset
data = pd.read_csv('titanic_data.csv')
# Print the first few entries of the Titanic data
data.head()


# In[55]:


# define variables(features, outcomes)
#Note: do not include Name column with features 
outcome = data[['Survived']]
features = data.drop(['Survived'],axis=1)

# Show the new dataset with 'Survived' removed
features.head()


# In[56]:


#data exploration:
print(data.info())
print(display(data))
print(features.isnull().sum())


# In[57]:


#data cleaning: fill null values with zero
features = features.fillna(0.0)
print(features.isnull().sum())
display(features.head(n=7))


# In[58]:


#additional point form me, here we can see the statistcal information about our data like
#like the one in last project for last lesson


# ## Preprocessing the data
# 

# In[59]:


#transformation: Perform feature scaling on the data
# first: define the standardization scaling object using StandardScaler().

stand = StandardScaler()
# second: apply the scaler to the numerical columns on the data:
numerical = ['PassengerId','Pclass','Age','SibSp','Parch','Fare']
features[numerical] = stand.fit_transform(features[numerical])
features.head()


# we'll one-hot encode the features.

# In[60]:


#dummies variables: convert catogrical columns to numerical
## perform one-hot encoding on categorical columns Using pandas.get_dummies()
features_final = pd.get_dummies(features)
features_final.head()


# In[61]:


features_final.shape


# ## Training the model
# 
# Now we're ready to train a model in sklearn. First, let's split the data into training and testing sets. Then we'll train the model on the training set.

# In[62]:


#split the data to two sets. training set and testing set:
X_train, X_test,y_train, y_test = train_test_split(features_final,outcome)


# In[63]:


# Define the classifier model as DecisionTree
clf = DecisionTreeClassifier()

#fit the model to the data
clf.fit(X_train, y_train)


# ## Testing the model
# Now, let's see how our model does, let's calculate the accuracy over both the training and the testing set.

# In[64]:


# Making predictions on scaling data
train_pred = clf.predict(X_train)
test_pred = clf.predict(X_test)

train_accuracy = accuracy_score(y_train,train_pred)
test_accuracy = accuracy_score(y_test,test_pred)

print('The training accuracy is', train_accuracy)
print('The test accuracy is', test_accuracy)


# # Improving the model
# 
# Ok, high training accuracy and a lower testing accuracy. We may be overfitting a bit.
# 
# So now it's your turn to shine! Train a new model, and try to specify some parameters in order to improve the testing accuracy, such as:
# - `max_depth` The maximum number of levels in the tree.
# - `min_samples_leaf` The minimum number of samples allowed in a leaf.
# - `min_samples_split` The minimum number of samples required to split an internal node.
# 
# 
# 
# use Grid Search!
# 
# 

# In[70]:


#grid search
#import gridsearch

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

#define the classifier model by DecisionTree
clf = DecisionTreeClassifier()

#define the parameters:
# HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}
parameters = {'max_depth':[3,6],'min_samples_leaf':[20,25],'min_samples_split':[20,25]}

#define the score method using make_scorer()
scorer = make_scorer(accuracy_score)

#define gridsearchcv function with cv=3 (so cross validation=3)
grid_obj = GridSearchCV(clf,parameters,scoring=scorer,n_jobs=-1,cv=3)
#fit/ train the function/ object
grid_fit = grid_obj.fit(X_train,y_train)
#get the best estimtor model
best_clf = grid_fit.best_estimator_


# In[71]:


# Make predictions using the new model.
y_train_pred = best_clf.predict(X_train)
y_test_pred = best_clf.predict(X_test)

# Calculating accuracies
train_accuracy = accuracy_score(y_train,y_train_pred)
test_accuracy = accuracy_score(y_test,y_test_pred)

print('The training accuracy is', train_accuracy)
print('The test accuracy is', test_accuracy)


# In[ ]:




