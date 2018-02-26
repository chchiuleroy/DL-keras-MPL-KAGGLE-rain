# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:13:17 2018

@author: roy
"""

import numpy as np; import pandas as pd
from joblib import Parallel, delayed; import multiprocessing

rawdata = pd.read_csv('C:/Users/roy/Documents/z_dataset/seattleWeather_1948-2017.csv')

def dl(rawdata):
    
    str = rawdata['DATE']
    str_sp = []
    for i1 in range(len(str)):
        str_sp.append(str[i1].split('-'))
    date_cov = pd.DataFrame(str_sp, columns = ['year', 'month', 'day'])
    rawdata_new = pd.concat([date_cov, rawdata.iloc[:, 1:]], axis = 1)
    
    id_na = rawdata_new.isnull().sum().astype('int'); ind = np.where(id_na>0)
    prcp_n = rawdata_new.iloc[:, ind[0][0]].dropna(axis = 0, how = 'any')
    hist, bin_n = np.histogram(prcp_n, density = True); prob = hist*np.diff(bin_n); prob.sum()
    choose_bin = np.random.choice(range(len(hist)), size = 1, p = prob)
    l1 = np.matrix(np.where(rawdata_new.iloc[:, ind[0][0]].isnull())).T
    for j1 in range(np.shape(l1)[0]):
        rawdata_new.iloc[l1[j1], ind[0][0]] = np.random.uniform(bin_n[choose_bin], bin_n[choose_bin+1], size = 1)
    rawdata_new.iloc[:, ind[0][0]].isnull().sum()
    
    rain_n = rawdata_new.iloc[:, ind[0][1]].dropna(axis = 0, how = 'any')
    level, freq = np.unique(rain_n, return_counts = True); prob1 = freq/freq.sum()
    l2 = np.matrix(np.where(rawdata_new.iloc[:, ind[0][1]].isnull())).T
    for j2 in range(np.shape(l2)[0]):
        rawdata_new.iloc[l2[j2], ind[0][1]] = np.random.choice(level, size = 1, p = prob1)
    rawdata_new.iloc[:, ind[0][1]].isnull().sum()
    
    rawdata_new.iloc[:, ind[0][1]] = rawdata_new.iloc[:, ind[0][1]].astype('int')
    
    ratio = 0.7; size = rawdata_new.shape[0]
    
    spl = int(ratio*size)
    id1 = np.random.choice(range(size), spl, replace = False)
    
    train = rawdata_new.iloc[id1, :]; test = rawdata_new.iloc[~id1, :]
    id2 = rawdata_new.shape[1]-1
    train_y = train.iloc[:, id2]; test_y = test.iloc[:, id2]
    train_x = train.iloc[:, range(id2)]; test_x = test.iloc[:, range(id2)]
    
    from keras.models import Sequential as seq
    from keras.layers.core import Activation, Dropout, Dense
    
    model = seq()
    model.add(Dense(units = 300, input_dim = 6, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dropout(0.3))
#    model.add(Dense(units = 5, input_dim = 6, kernel_initializer = 'uniform', activation = 'relu'))
#    model.add(Dropout(0.3))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.fit(np.array(train_x), np.array(train_y), epochs = 50, batch_size = 80, validation_split = 0.3, verbose = 2)
#    model.fit(np.array(train_x), np.array(train_y), epochs = 50, batch_size = 80, validation_split = 0.3, verbose = 1)
    train_acc = model.evaluate(np.array(train_x), np.array(train_y))
    test_acc = model.evaluate(np.array(test_x), np.array(test_y))
    results = [train_acc[1], test_acc[1]]
    return results

dl(rawdata)

runs = 30
r = []
for l in range(runs):
    r.append(dl(rawdata))