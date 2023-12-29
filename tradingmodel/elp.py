import pandas as pd
import tensorflow as tf
import numpy as np
import copy
import time
import ccxt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from checking import checking
from components.fetchdata import fetchdata, filt
from components.scaller import scaller

bitmex   = ccxt.bitmex({
    'apiKey': 'rHtA_bklMiwfmlRuJIkWthFt',
    'secret': 'i24DSLItVmj6kPAvEO2vu_dieK6Rn1VdLpUAaYhcrWb7Nlcj',
})
bitmex.urls['api'] = bitmex.urls['test']
model_high = tf.keras.models.load_model('./models/tradingmodel5mhigh')
model_down = tf.keras.models.load_model('./models/tradingmodel5mdown')

def makePredict():

    noww = datetime.now()
    print(noww)
    XXX = fetchdata(noww,1,30)
    XXX=np.array([np.array(x) for x in XXX])
    scaler = MinMaxScaler()
    XXX = scaler.fit_transform(XXX)
    XXX = np.array([XXX])
    XXX = tf.constant(XXX)
    y_pred_down = np.array([x[0] for x in model_down.predict_classes(XXX)])[0]
    y_pred_high = np.array([x[0] for x in model_high.predict_classes(XXX)])[0]
    print('down predict is  : ', y_pred_down, '   high predict is  : ', y_pred_high)
    return y_pred_high, y_pred_down

res = pd.DataFrame(columns=['Date', 'Close', 'Open', 'High', 'Low','Predict_high', 'Predict_down', 'Move','High-Low'])
res.to_csv('results.csv')
n = 0
while True:
        if datetime.now().strftime("%S") == '00':
            date = datetime.now()
            y_pred_high, y_pred_down = makePredict()
            if y_pred_high==1 and y_pred_down==0:
                a = 1
            elif y_pred_high==0 and y_pred_down==1:
                a = 0
            else:
                a = 2
            cc = bitmex.fetchTickers() 
    # print(cc)
            cc = pd.DataFrame(cc) 
            date = datetime.now()
            high = cc['BTC/USD'].high 
            low = cc['BTC/USD'].low 
            close = cc['BTC/USD'].close
            openn = cc['BTC/USD'].open
            rescsv = pd.read_csv('./results.csv')
            rescsv.pop('Unnamed: 0')
            # print(rescsv)
            # print(rescsv.loc[0].open)
            try:
                if openn<close:
                    b = 1
                elif openn>close:
                    b=0
                else:
                    b=2
            except:
                b='nothing'
            
            rescsv.loc[n]=[date,close, openn, high, low, y_pred_high, y_pred_down, a, b]
            rescsv.to_csv('results.csv')
            print(rescsv)
            time.sleep(55)
            n = n+1
        

