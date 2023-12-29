import pandas as pd
import time


import datetime
import numpy as np
import copy
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import KFold
import copy
from decimal import Decimal

# get market info for bitcoin from the start of 2016 to the current day



import time
import os
import pandas as pd
import numpy as np
import ccxt
exchange = ccxt.bitmex({
    'rateLimit': 10000,
    'enableRateLimit': True,
    # 'verbose': True,
})
since1 = exchange.parse8601('2018-12-01T00:00:00Z')  # January

symbol = 'BTC/USD'
timeframe = '1h'
ohlcv1 = exchange.fetch_ohlcv(symbol, timeframe, since1,1000)

df = pd.DataFrame(ohlcv1, columns=['Date','Open','High','Low','Close','Volume'])
# Setting the datetime index as the date, only selecting the 'Close' column, then only the last 1000 closing prices.
from datetime import datetime
df['Date']=pd.to_datetime(df['Date']*1000000)
print(df)

symbol = 'ETH/USD'
timeframe = '1h'
ohlcv1 = exchange.fetch_ohlcv(symbol, timeframe, since1,1000)

dfbtc_eth = pd.DataFrame(ohlcv1, columns=['Date','Open','High','Low','Close','Volume'])
# Setting the datetime index as the date, only selecting the 'Close' column, then only the last 1000 closing prices.
from datetime import datetime
dfbtc_eth['Date']=pd.to_datetime(dfbtc_eth['Date']*1000000)



eurusdtXopen = dfbtc_eth[dfbtc_eth.columns[1]].values.astype('double')
eurusdtXhigh = dfbtc_eth[dfbtc_eth.columns[2]].values.astype('double')
eurusdtXlow = dfbtc_eth[dfbtc_eth.columns[3]].values.astype('double')
eurusdtXclose = dfbtc_eth[dfbtc_eth.columns[4]].values.astype('double')
eurusdtXvolume = dfbtc_eth[dfbtc_eth.columns[5]].values.astype('double')


 



Xopen = df[df.columns[1]].values.astype('double')
Xhigh = df[df.columns[2]].values.astype('double')
Xlow = df[df.columns[3]].values.astype('double')
Xclose = df[df.columns[4]].values.astype('double')
Xvolume = df[df.columns[5]].values.astype('double')

N = 100000
y = df[df.columns[2]].values.tolist()

XXX=[]
Y= []


a1 = []

#iter=1119
#n=30

iter = 950
n=30
interval = 5

print(type(Xopen))

rr=[]

for i in range(iter):
   XXX.append(np.concatenate((Xopen[range(i,i+n)], 
   Xhigh[range(i,i+n)],
   Xlow[range(i,i+n)], 

   Xclose[range(i,i+n)],
   Xvolume[range(i,i+n)]
                              ,eurusdtXopen[range(i,i+n)],eurusdtXhigh[range(i,i+n)],eurusdtXlow[range(i,i+n)],eurusdtXclose[range(i,i+n)],eurusdtXvolume[range(i,i+n)]
                              ), axis=0))
   
   
   #print(range(i,i+n))
   #print(i)
   #print(i+n)
   #print(i+n-1)
   Y.append(y[i+n+interval]/Xclose[i+n]>1.005 or y[i+n-1+interval]/Xclose[i+n]>1.005 or y[i+n-2+interval]/Xclose[i+n]>1.005
   )
print(len(Y))
print(sum(Y))


XXX=np.array(XXX)

print(Y)
Y=np.array(Y)



from operator import add



from autoPyTorch import AutoNetClassification

# data and metric imports
import sklearn.datasets
import sklearn.metrics




a1 = []
f1 = []
ff1=[]

kf = KFold(n_splits=10)

step=24
iterations=5
learn=800
i=-step
while i<iterations*step+1:
       i=i+step
       print("i,learn+i-1",i,learn+i-1)
       print("learn+i:learn+i+step",learn+i,learn+i+step)    
       X_train, X_test = XXX[i:learn+i-1], XXX[learn+i:learn+i+step]
       y_train, y_test = Y[i:learn+i-1],Y[learn+i:learn+i+step]
       print(y_train,y_test)

# running Auto-PyTorch
       autoPyTorch = AutoNetClassification("tiny_cs", budget_type="epochs", # config preset
                                    log_level='info',
                                   
                                    min_budget=30,
                                    max_budget=90,
                                    num_iterations=1,cuda=True)
                                    #networks=["resnet", "shapedresnet", "mlpnet", "shapedmlpnet"])



       autoPyTorch.fit(X_train, y_train)
       

       y_pred = autoPyTorch.predict(X_test)
       print("y_test",y_test)
       print("y_pred",y_pred)
       f1_1=f1_score(y_test, y_pred)
       a1_1=sklearn.metrics.accuracy_score(y_test, y_pred)
       ii=0
       cont_true=0
       cont_true_true=0
       for a in y_pred:
           print(a[0])
           
           if a[0]==1 :
               cont_true=cont_true+1
               if y_test[ii]:
                   cont_true_true=cont_true_true+1
           ii=ii+1
       if cont_true>0:
           ff1_1=cont_true_true/cont_true
       else:
           ff1_1=1

       print("ff1",ff1_1)
       ff1.append(ff1_1)
       print("Accuracy score",a1_1)
       print("f1",f1_1)
       a1.append(a1_1)
       f1.append(f1_1)
print(a1)
print(f1)
print(np.average(a1))
print(np.average(f1))
print("ff1",ff1)
print("ff1",np.average(ff1))