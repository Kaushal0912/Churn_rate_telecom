# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 21:09:27 2020

@author: lenovo
"""
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

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

 

