# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:53:36 2023

@author: user
"""

import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score



df=pd.read_csv('data (2).csv')
#Data Preprocessing
df.drop(["id","Unnamed: 32"],axis=1,inplace=True)
df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]
y = df[['diagnosis']]
x = df.drop(["diagnosis"],axis=1)
x=(x-np.min(x))/(np.max(x)-np.min(x))
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)


def sigmoid(z):
    return (1/(1+np.exp(-z)))   
    
def forward_propagation(x_train,weight,bias,y_train):
    z=np.dot(weight.T,x_train)+bias
    a=sigmoid(z)
    cost=-((np.log(a)*y_train.T.values)+(np.log(1-a)*(1-(y_train.T.values)))).sum()/x_train.shape[1]
    return cost,a
def backward_popagation(a_,x_train,y_train,alfa,weight,bias):
    d_weight=np.sum((a_-y_train.T.values)*(x_train.values),axis=1).reshape(-1,1)/(x_train.shape[1])
    weight-=alfa*d_weight
    d_bias=((a_-y_train.T.values).sum())/(x_train.shape[1])
    bias-=alfa*d_bias
    return weight,bias
def logistic_regression(x_train,x_test,y_train,y_test,iteration,alfa):
    y_head=[]
    x_train=x_train.T
    x_test=x_test.T
    weight=np.full(x_train.shape[0],0.01).reshape(-1,1)
    bias=0.01
    cost_list=[]
    while True:
        cost_,a_=forward_propagation(x_train,weight,bias,y_train)
        cost_list.append(cost_)
        weight,bias=backward_popagation(a_,x_train,y_train,alfa,weight,bias)
        iteration-=1
        if iteration<=0:
            break
    
    z_=np.dot(weight.T,x_test)+bias
    a_1=sigmoid(z_)
    print('a')
    print(a_1)
    y_head=(a_1>=0.52).astype(int).reshape(-1,1)
    return y_head

#logistic_regression
y_pro=logistic_regression(x_train,x_test,y_train,y_test,45,0.0009)
print(accuracy_score(y_test, y_pro))
f1 = f1_score(y_test, y_pro)
print("F1 Skoru:", f1)



    
    
    











