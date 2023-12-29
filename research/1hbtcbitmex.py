import pandas as pd
import time
import datetime
import numpy as np
import copy
import pandas as pd
from decimal import Decimal
# get market info for bitcoin from the start of 2016 to the current day
import os
import ccxt
from operator import add
from autoPyTorch import AutoNetClassification
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import KFold
import sklearn.datasets
import sklearn.metrics
from sklearn.metrics import f1_score, accuracy_score
from Struct import Struct
from finta import TA
import math
from autoPyTorch import AutoNetRegression
exchange = ccxt.bitmex({
    'rateLimit': 10000,
    'enableRateLimit': True,
    # 'verbose': True,
})
since1 = exchange.parse8601('2018-12-01T00:00:00Z')  # January

def LoadCSVFromGDrive(fileKey):
    data = pd.read_csv('https://docs.google.com/spreadsheets/d/{}/gviz/tq?tqx=out:csv&sheet=XBUSD_1hour'.format(fileKey))  
    #data.to_csv(fileName,index=False)
    return data

def DataFrameToColums(dataFrame):
    opens = dataFrame[dataFrame.columns[1]].iloc[0:].values.astype('double') 
    highs = dataFrame[dataFrame.columns[2]].iloc[0:].values.astype('double') 
    lows = dataFrame[dataFrame.columns[3]].iloc[0:].values.astype('double') 
    closes = dataFrame[dataFrame.columns[4]].iloc[0:].values.astype('double') 
    volumes = dataFrame[dataFrame.columns[5]].iloc[0:].values.astype('double') 

    return Struct(opens=opens,highs=highs,lows=lows,closes=closes, volumes=volumes, OHLCV=dataFrame)

def TAIndicators(dataFrame):
    rsi=TA.RSI(dataFrame)/100
    #cci=TA.CCI(dataFrame)
    #mom=TA.MOM(dataFrame)
    #macd=TA.MACD(dataFrame)
    #WILLIAMS= TA.WILLIAMS(dataFrame)
   # AO=TA.AO(dataFrame)

    
    ema5=TA.EMA(dataFrame,5)/dataFrame['close']-1 #ema 100, current 105, we are above ema,
    ema10=TA.EMA(dataFrame,10)/dataFrame['close']-1
    ema20=TA.EMA(dataFrame,20)/dataFrame['close']-1
    ema30=TA.EMA(dataFrame,30)/dataFrame['close']-1
    ema50=TA.EMA(dataFrame,50)/dataFrame['close']-1
    ema100=TA.EMA(dataFrame,100)/dataFrame['close']-1
    ema200=TA.EMA(dataFrame,200)/dataFrame['close']-1

    ma5=TA.SMA(dataFrame,5)/dataFrame['close']-1
    ma10=TA.SMA(dataFrame,10)/dataFrame['close']-1
    ma20=TA.SMA(dataFrame,20)/dataFrame['close']-1
    ma30=TA.SMA(dataFrame,30)/dataFrame['close']-1
    ma50=TA.SMA(dataFrame,50)/dataFrame['close']-1
    ma100=TA.SMA(dataFrame,100)/dataFrame['close']-1
    ma200=TA.SMA(dataFrame,200)/dataFrame['close']-1

   # vwma=TA.WMA(dataFrame,20,"volume")/dataFrame['volume']

    return [rsi,ema5,ema10,ema20,ema30,ema50,ema100,ema200,ma5,ma10,ma20,ma30,ma50,ma100,ma200]

#-1 sell
#1 buy
#0 neural
def DecisionBasedOnTA(dataFrame,index):
    rsiValue=dataFrame.RSI[index]
    rsi=0
    if rsiValue>=70: 
        rsi=-1
    if rsiValue<=30: 
        rsi=1

    return [rsi]


def GetSymbolData(symbol,timeframe ):
    ohlcv1 = exchange.fetch_ohlcv(symbol, timeframe, since1,1000)
    df = pd.DataFrame(ohlcv1, columns=['Date','Open','High','Low','Close','Volume'])
    df['Date']=pd.to_datetime(df['Date']*1000000)
    print(df)
    r=DataFrameToColums(df)
    return r

def IsSuccessCondition(data1,lastIndex):
    


    betterForPerc=0.05/100
    nextNSkipToOutput=0
    nextNTakeToOutput=1
    rFrom=nextNSkipToOutput+1#3
    rTo=nextNSkipToOutput+nextNTakeToOutput+1#6
    highs=data1.high

    success=False
    currentClose=data1.close[lastIndex]
    nextClose=data1.close[lastIndex+1]
    diff=nextClose/currentClose-1
    #for x in range(rFrom, rTo):
   #     nextHigh=highs[lastIndex+x]
    #    diff=nextHigh/currentClose
     #   ok=diff>betterForPerc+1
     #   success=success or ok
    return diff

def GetTrainData(data1,indicators):
    count=438000
    input = [] 
    ouput= []
    
    for i in range(320000,count):
        #O, O, O, O, O
        #H, H, H, H, H
        #...                 ====> O,O,O,O,O,H, H, H, H, H...., V, V, V, V, V
        #V, V, V, V, V
        block=[indicators[0][i], 
            indicators[1][i], 
            indicators[2][i], 
               indicators[3][i], 
                 indicators[4][i], 
                   indicators[5][i], 
                     indicators[6][i], 
                       indicators[7][i], 
                         indicators[8][i], 
                           indicators[9][i], 
                              indicators[10][i], 
                      
                                    indicators[11][i], 
                                    indicators[12][i], 
                                        indicators[13][i], 
                                        indicators[14][i], 
                                       # indicators[i][15]
           ]

        input.append(block)
        indexOfLastElement=i
        val=IsSuccessCondition(data1,indexOfLastElement)
        #decVal=(0.0,1.0)[val]
        ouput.append(val)

    return Struct(Input=input,Output=ouput)          

btcusd5min=LoadCSVFromGDrive("154rDDWpKqPO1XReY0mAI4MsGu2m4medqmOq1jYP9MNQ")
indicators=TAIndicators(btcusd5min)

#btcusd=GetSymbolData('BTC/USD','1h')
#ethusd=GetSymbolData('ETH/USD','1h')


trainData=GetTrainData(btcusd5min,indicators)


print('Data lines: %',len(trainData.Output))
print('Positive cases: %',sum(trainData.Output))

print(trainData.Output)

#===========Not refactored============================================
a1 = []
f1 = []
ff1=[]

kf = KFold(n_splits=10)


iterations=5
learn=50000
testOn=200
totalTests=len(trainData.Input)
testsInOneIteration=learn+testOn
totalIterations=math.floor(totalTests/testsInOneIteration)

for i in range(0,totalIterations):
       startLearn=testsInOneIteration*i
       endLearn=startLearn+learn-1
       startTest=endLearn+1
       endTest=startTest+testOn-1

       print("learn range:",startLearn,endLearn)
       print("test range:",startTest,endTest) 
       X_train, X_test = trainData.Input[startLearn:endLearn], trainData.Input[startTest:endTest]
       y_train, y_test = trainData.Output[startLearn:endLearn],trainData.Output[startTest:endTest]
       print(y_train,y_test)

# running Auto-PyTorch
       autoPyTorch =  AutoNetClassification("tiny_cs", budget_type="epochs", # config preset
                                    log_level='debug',
                                   
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
