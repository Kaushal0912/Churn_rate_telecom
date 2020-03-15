# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 21:09:27 2020

@author: lenovo
"""
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFE

dataset  = pd.read_excel('C:/Users/lenovo/Desktop/DOCUMENTS/AcmeTeleABT.xlsx') 
dataset.drop(dataset.columns[-1],axis=1,inplace=True)
dataset.drop(['occupation','regionType'],axis=1,inplace=True)
encoded_ds = copy.deepcopy(dataset)
X_train = encoded_ds['churn']
encoded_ds.drop('churn',axis=1,inplace=True)
label_encoder = LabelEncoder()
integer_encoded_children = label_encoder.fit_transform(dataset['children'])
integer_encoded_credit = label_encoder.fit_transform(dataset['credit'])
integer_encoded_creditCard = label_encoder.fit_transform(dataset['creditCard'])
integer_encoded_marry = label_encoder.fit_transform(dataset['marry'])
integer_encoded_churn = label_encoder.fit_transform(dataset['churn'])
encoded_ds['children'] = integer_encoded_children
encoded_ds['credit'] = integer_encoded_credit
encoded_ds['creditCard'] = integer_encoded_creditCard
encoded_ds['marry'] = integer_encoded_marry
for col in encoded_ds.columns:
    print('\n')
    print('************Stats of '+col+'****************')
    print(encoded_ds[col].describe())
    plt.hist(encoded_ds[col])
    plt.show()   
    print('************End of stats for '+col+'*************')
    print('\n')

 
    
    
#build X and y
X = encoded_ds.iloc[:,0:-1] #all columns except last column
y = integer_encoded_churn 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)    
clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_rf = RandomForestClassifier(random_state=1, n_estimators=100, max_depth=100, max_leaf_nodes=100)
clf_lr = LogisticRegression(class_weight="balanced")
estimators = [('knn', clf_knn), ('lr', clf_lr), ('dt', clf_rf)]
clf_avg = VotingClassifier(estimators,voting='soft')
clf_avg.fit(X_train,y_train)
print(accuracy_score(y_test, clf_avg.predict(X_test)))
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
print(accuracy_score(y_true, y_pred))


plt.figure(figsize=(12,10))
cor = encoded_ds.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
