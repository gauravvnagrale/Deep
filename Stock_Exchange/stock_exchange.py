# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 12:06:42 2018

@author: HARSH
"""
import numpy as np
import pandas as pd
from sklearn import model_selection
import matplotlib.pyplot as plt

def architecture(L,non):
    w = []
    b = []
    for i in range(L):
        w.append(np.random.randn(non[i+1],non[i]))
        b.append(np.random.randn(non[i+1]).reshape((non[i+1],1)))
    return w,b


def sigmoid(z):
    return 1/(1+np.exp(-z))


def ReLU(z):
    z[z<=0] = 0
    return z


def relu_derivative(z):
    z[z>0] = 1
    z[z<=0] = 0
    return z


def forward_propagate(X,w,b,L):
    a = []
    z = []
    a.append(X)
    z.append(X)
    for i in range(L):
        #print(a[-1].shape,w[i].shape,b[i].shape)
        z.append(np.dot(a[-1] , w[i].T) + b[i].T)
        a.append(ReLU(z[-1]))
    return a,z


def back_propagation(a,z,w,b,L,y):
    m = y.shape[0]
    da = []
    dw = []
    dz = []
    db = []
    
    cost = (1/m)*np.sum(np.square(a[-1]-y))
    
    dz.append(a[-1] - y)
    dw.append(np.dot(dz[-1].T,a[L-1]))
    db.append((np.sum(dz[-1],axis = 0).T).values.reshape((dz[-1].shape[1],1)))
    
    for i in range(L-1):
        da.append(np.dot(dz[-1],w[L-1-i]))
        dz.append(da[-1]*relu_derivative(z[L-1-i]))
        dw.append(np.dot(dz[-1].T , a[L-2-i]))
        db.append(np.sum(dz[-1],axis = 0).T.reshape((dz[-1].shape[1],1)))
    
    grads = {"dw" : dw,
             "db" : db}
    return cost,grads


def update(w,b,grad,L,l_rate,m,lambd):
    dw = grad["dw"]
    db = grad["db"]
    
    for i in range(L):
        w[i] = w[i] - (l_rate/m)*(dw[L-1-i] + (lambd*w[i]))
        b[i] = b[i] - (l_rate/m)*(db[L-1-i] + (lambd*b[i]))
    
    return w,b


def predict(X,w,b,L):
    a,z = forward_propagate(X,w,b,L)
    return a[-1]


def gradient_desc(X,y,w,b,l_rate,n_iter,L,lambd,X_test,y_test):
    m = X.shape[0]
    costtrain = []
    costtest = []
    for i in range(n_iter):
        a,z = forward_propagate(X,w,b,L)
        cost,grad = back_propagation(a,z,w,b,L,y)
        w,b = update(w,b,grad,L,l_rate,m,lambd)
        if(i%100==0):
            costtrain.append(cost)
            a,z = forward_propagate(X_test,w,b,L)
            costtest.append((1/X_test.shape[0])*np.sum(np.square(a[-1]-y_test)))
            print(costtrain[-1]," ",costtest[-1])
    return w,b,costtrain,costtest



#main code
data = pd.read_csv('data.csv')
data = data.iloc[0:2538,0:5]

#converting datatypes
lst = ['Price','Open','High','Low','y']
for j in range(len(lst)):
    var = data[lst[j]]
    var = var.astype(str)
    a = []
    for i in var:
        a.append(i.split(','))
    b = []
    for i in range(len(a)):
        if(len(a[i])>1):
            b.append(a[i][0]+a[i][1])
        else:
            b.append(a[i][-1])
    a = pd.Series(b)
    a = a.astype(float)
    data[lst[j]] = a
    

X = data.iloc[:,0:4]
#Preprocessing
X = X/np.linalg.norm(X,axis=0)
y = data.iloc[:,4:5]
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.2)
#error = <600
#l_rate = 1.5
#n_iter = 50000
#lambd = 0.01
l_rate = 1.38
n_iter = 50000
L = 1
lambd = 0.00001
non = [X.shape[1],1]
w,b = architecture(L,non)
w,b,costtrain,costtest = gradient_desc(X_train,y_train,w,b,l_rate,n_iter,L,lambd,X_test,y_test)
y_pred = predict(X_test,w,b,L)
#acc = np.sum(np.square(y_pred - y_test))
#y_pred = predict(X_train,w,b,L)
#acc1 = np.sum(np.square(y_pred - y_train))
#print("L_rate = ",l_rate)
#print("1.test set = ",acc," 2.training set = ",acc1)
plt.plot(costtrain)
plt.plot(costtest)
plt.show()

