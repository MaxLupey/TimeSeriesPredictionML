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
    # print('111111111',XXX)
    # df = fetchData('ETH/USD', 1, '5m', 30, 'ETHUSD5.csv', 5, noww)
    # edf = fetchData('BTC/USD', 1, '5m', 30, 'BTCUSD5.csv', 5, noww)
    # df = filt(df)
    # edf = filt(edf)
    # XXX = np.concatenate((df,edf), axis=1)
    # XXX = np.expand_dims(XXX, axis=0)
    # XXX = XXX.astype(np.float32)
    XXX=np.array([np.array(x) for x in XXX])
    # XXX=XXX.astype(int)
    scaler = MinMaxScaler()
    XXX = scaler.fit_transform(XXX)
    XXX = np.array([XXX])
    XXX = tf.constant(XXX)
    y_pred_down = np.array([x[0] for x in model_down.predict_classes(XXX)])[0]
    y_pred_high = np.array([x[0] for x in model_high.predict_classes(XXX)])[0]
    print('down predict is  : ', y_pred_down, '   high predict is  : ', y_pred_high)
    return y_pred_high, y_pred_down

def fetchOrders(): 
  openOrders = len(bitmex.fetch_open_orders()) 
  isLessOrders = openOrders<2 
  # print('LimitOrders: ', openOrders) 
  return isLessOrders, openOrders 

def showTickers(): 
  cc = bitmex.fetchTickers() 
  cc = pd.DataFrame(cc) 
  cc1 = cc['BTC/USD'].bid 
  cc2 = cc['BTC/USD'].ask 
  print('CC',cc1, cc2) 
  hight = round(cc2*1.001, 2) 
  down = round(cc1*0.999, 2) 
  print('H/D :',hight, down) 
  return hight, down 

def createSellLimits(): 
  hight, down = showTickers() 
  cc = bitmex.fetchTickers() 
  cc = pd.DataFrame(cc) 
  cc1 = cc['BTC/USD'].bid 
  cc2 = cc['BTC/USD'].ask 
  symbol = 'BTC/USD'  
  typee = 'StopLimit'  
  side = 'sell' 
  amount = 250
  price = int(round(cc1*0.995, 2))  
  params = { 
    'stopPx': int(round(down*0.999, 2)),  
  } 
  bitmex.create_order(symbol, typee, side, amount, price, params) 
  bitmex.create_limit_sell_order('BTC/USD', 250, int(hight)) 
  print('Limits are created') 

def createBuyLimits(): 
  cc = bitmex.fetchTickers() 
  cc = pd.DataFrame(cc) 
  cc1 = cc['BTC/USD'].bid 
  cc2 = cc['BTC/USD'].ask 
  hight, down = showTickers() 
  symbol = 'BTC/USD'  
  typee = 'StopLimit'  
  side = 'buy' 
  amount = 250
  price = int(round(cc2*1.005, 2))  
  params = { 
    'stopPx': int(round(hight*1.001, 2)),  
  } 
  bitmex.create_order(symbol, typee, side, amount, price, params) 
  bitmex.create_limit_buy_order('BTC/USD', 250, int(hight)) 
  print('Limits are created') 

def watching():
  cc = bitmex.fetchTickers() 
  # print(cc)
  cc = pd.DataFrame(cc) 
  date = datetime.now()
  print(date, '  -  ', cc['BTC/USD'].open, '       ', cc['BTC/USD'].close)
  # cc1 = cc['BTC/USD'].bid 
  # cc2 = cc['BTC/USD'].ask 
  return cc['BTC/USD'].open, cc['BTC/USD'].close
    
n=0
preds = []
ifend = False
report = pd.DataFrame(columns=['Date', 'Start', 'End', 'Predict_high', 'Predict_down', 'Real'])
while True:
  if datetime.now().strftime("%S") == '00':
    try: 
      print('--------------'+str(n)+'-----------')
      # пауза на 5 сек
      time.sleep(5)
      # зафіксувати перший клоуз і дату
      _,openn = watching()
      date = datetime.now()
      # зробити предікт
      y_pred_high, y_pred_down = makePredict()
      preds.append([y_pred_high, y_pred_down])
      # якшо не початок і не менше ордерів робити перенвірку кожні 30 сек
      if not n==0 and ifend==False:
        for i in range(9):
          print(i,'..........=>')
          ifend = checking()  
          time.sleep(30)
          # якшо вибило зупинити перевірки і зробити нові дії
          if ifend==True:
            print('breaking')
            # зробити новий предікт 
            y_pred_high, y_pred_down = makePredict()
            preds.append([y_pred_high, y_pred_down])
            # зафіксувати новий перший клоуз і дату
            _,openn = watching()
            date = datetime.now()
            break
        time.sleep(15)
      print('11111111111111')
      # якшо початок або вибило не робити перевірок а почекати до завершення 5 хвилин
      if n==0:
        time.sleep(290)
      elif ifend==True:
        time.sleep(275)
      print('2222222222222222')
      isLessOrders, _ = fetchOrders()
      # зробити дії
      if y_pred_high==1 and y_pred_down==0:
        if n==0 or isLessOrders:
          bitmex.cancel_all_orders()
          bitmex.create_market_buy_order('BTC/USD',250)
          createSellLimits()
          changeifend=True
          print('buy')
      elif y_pred_high==0 and y_pred_down==1:
        if n==0 or isLessOrders:
          bitmex.cancel_all_orders()
          bitmex.create_market_sell_order('BTC/USD', 250)
          createBuyLimits()
          changeifend=True
          print('sell')
      else:
        # якшо початок і бот нічого не предсказав - повторити дію, доки не предскаже
        if n==0:
          n=-1
        print('bot say no move')
      print('3333333333')
      # взяти клоуз на кінці
      _, close = watching()
      # визначити реальне значення
      if openn-close<0:
        res = 1
      elif openn-close>0:
        res = 0
      else :
        res = 2
      # додати дані в репорт
      print('44444444444')
      if not n==0 and ifend==False:
        report.loc[n] = [date, openn, close, preds[n-1][0], preds[n-1][1], res]
      # якшо був брефкінг не ставити значення предікту
      if not n==0 and ifend==True:
        report.loc[n] = [date, openn, close, 'breaking', 'breaking', res]
        # якшо зробилася дія в трейді, то вернути іфенд в фолс шоб далі перевіряти кожні 30 сек
        if changeifend==True:
          ifend=False
      print(report)

      print('-----------end.......=>--------------')
      n=n+1
      
    except Exception as e: 
      print(e) 
      print('somerthing whent wrong!!!!!!!')