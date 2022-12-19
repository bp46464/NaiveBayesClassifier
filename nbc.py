# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 22:45:57 2021

@author: Przemysław
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import datasets
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
import numpy as np


class nbc(BaseEstimator, ClassifierMixin):
    
    def __init__(self):
        self.X = None
        self.y = None
        self.apriori = None
        
    def fit(self, _X, _y, _ap):
        self.X = _X
        self.y = _y
        self.apriori = _ap;
        
    def predict(self):
        xy,yy = self.X.shape
            
        #będą przechowywały prawdopodobieństwa warunkowe
        y0 = []
        y1 = []
        y2 = []

        #w zasadzie robienie P(A and B)
        for j in range(xy):
            counter00 = 0
            counter01 = 0
            counter02 = 0
            counter10 = 0
            counter11 = 0
            counter12 = 0
            counter20 = 0
            counter21 = 0
            counter22 = 0
            for i in range(yy):
                if self.y[j] == 0: #jeżeli przewidziane y to 0
                    if self.X[j][i] == 0:
                        counter00=counter00+1
                    if self.X[j][i] == 1:
                        counter01=counter01+1
                    if self.X[j][i] == 2:
                        counter02=counter02+1
                if self.y[j] == 1:
                    if self.X[j][i] == 0:
                        counter10=counter10+1
                    if self.X[j][i] == 1:
                        counter11=counter11+1
                    if self.X[j][i] == 2:
                        counter12=counter12+1
                if self.y[j] == 2:
                    if self.X[j][i] == 0:
                        counter20=counter20+1
                    if self.X[j][i] == 1:
                        counter21=counter21+1
                    if self.X[j][i] == 2:
                        counter22=counter22+1
            #dodanie prawdopodobieństw        
            y0.append(counter00)
            y0.append(counter01)
            y0.append(counter02)
            y1.append(counter10)
            y1.append(counter11)
            y1.append(counter12)
            y2.append(counter20)
            y2.append(counter21)
            y2.append(counter22)
        #zrobienie z nich tablic o wymiarach (X x 3)
        rows = int(len(y0)/3)
        y0 = np.array(y0)
        y0 = y0.astype(float)
        y0 = y0.reshape((rows,3))
        y1 = np.array(y1)
        y1 = y1.astype(float)
        y1 = y1.reshape((rows,3))
        y2 = np.array(y2)
        y2 = y2.astype(float)
        y2 = y2.reshape((rows,3))
        
        #wykonanie dzielenia przez P(B)
        rx, ry = y0.shape
        for j in range(rx):
            for i in range(ry):
                y0[j][0] = y0[j][0] / apTab[0]
                y0[j][1] = y0[j][1] / apTab[1]
                y0[j][2] = y0[j][2] / apTab[2]
                y1[j][0] = y1[j][0] / apTab[0]
                y1[j][1] = y1[j][1] / apTab[1]
                y1[j][2] = y1[j][2] / apTab[2]
                y2[j][0] = y2[j][0] / apTab[0]
                y2[j][1] = y2[j][1] / apTab[1]
                y2[j][2] = y2[j][2] / apTab[2]
                
        #wykonanie wzoru 6.14
        moje_y = []            
        for j in range(xy):            
            s0 = y0[j][0] * y0[j][1] * y0[j][2] * (apTab[0] / len(y_test))
            s1 = y1[j][0] * y1[j][1] * y1[j][2] * (apTab[1] / len(y_test))
            s2 = y2[j][0] * y2[j][1] * y2[j][2] * (apTab[2] / len(y_test))
            
            #wstawianie wyników do listy
            if s0>s1 and s0>s2:
                moje_y.append(0)
            elif s1>s0 and s1>s2:
                moje_y.append(1)
            elif s2>s1 and s2>s0:
                moje_y.append(2)
            else: #czasami są remisy
                moje_y.append(100)      
        
        return np.array(moje_y)
    
    def score(self, y):
        score = 0.0
        for i in range(len(y)):
            if y[i] == self.y[i]:
                score = score+1
        return round(score/len(self.y)*100,2)


dataset = datasets.load_wine()
X = dataset.data
y = dataset.target
est = KBinsDiscretizer(n_bins=3, encode="ordinal")
xt = est.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(xt, y, train_size=0.95)

#dziedziny 
d = np.unique(y_train)
apTab = np.zeros((3,1))

#apriori
for i in range(len(d)):
    counter = 0
    for v in y_train:
        value = i
        if v == value:
            counter = counter+1
            
    apTab[i]=counter


bayes = nbc()
bayes.fit(X_train, y_train, apTab)
pred = bayes.predict()
wyn = bayes.score(pred)

print("Score: ", wyn,"%")
            