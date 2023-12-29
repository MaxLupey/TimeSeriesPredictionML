import tensorflow as tf
import copy
import time
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from components.scaller import scaller
from components.unifyclassessizes import unify_classes_sizes 
from components.createmodel import create_model
from components.createdataset import createdataset
from components.fetchdata import fetchdata


def creating(mode, name):
  d =  datetime.strptime('2021-01-14 00:00:00', "%Y-%m-%d %H:%M:%S")
  print(d) 
  # dataar, data, close, high = fetchdata(d)
  dataar = fetchdata(d,1000,300)
  
  # X, Y, dates, closes, highes = createdataset(dataar, data, close, high, mode)
  X, Y = createdataset(dataar, mode)
  print(X.shape, Y.shape)
  # X , X_test, Y , y_test, dates, dates_test, closes, closes_test = train_test_split(X, Y, dates, closes,  test_size=0.1, random_state=0, shuffle = False)
  X , X_test, Y , y_test = train_test_split(X, Y,  test_size=0.1, random_state=0, shuffle = False)
  print('00000 :',len([a  for a in Y if a==0]),',  1111 :', len([a for a in Y if a==1]) )
  print(X.shape, Y.shape)
  X, y = unify_classes_sizes(X,Y)
  print(X.shape, y.shape)
  X = scaller(X) 
  print(X)
  print(X.shape)

  X_train ,X_val, y_train, y_val = train_test_split(X, y,  test_size=0.33, random_state=42, shuffle = True)
  print(X_train.shape, y_train.shape)
  # model  = create_model(X_train.shape[1],X_train.shape[2],"accuracy") 

  # model.fit(X_train, y_train,batch_size=32,
  #       epochs=20,validation_data=(X_val, y_val))

  # y_pred =np.array([x[0] for x in model.predict_classes(X_val)])
  # cf_matrix = confusion_matrix(y_val, y_pred)
  # print(cf_matrix)


  # model.save(os.path.join('models', name))

creating('sell', 'tradingmodel5msell')
creating('buy', 'tradingmodel5mbuy')


