import ccxt
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def fetchData(name, number, timeframe, numb2, name2, n, date):
  noww = date

  print(noww)
  symbol = name
  exchange = ccxt.bitmex({
      'rateLimit': 2000,
      'enableRateLimit': True,
      # 'verbose': True,
  })
  df = pd.DataFrame(columns=['Date','Open','High','Low','Close','Volume'])
  for i in range(number):
    now = noww - timedelta(minutes=1000*i*n) 
    # print(now)
    string = now.strftime("%d-%m-%Y %H:%M:%S")
    since1 = exchange.parse8601(now)
    ohlcv1 = exchange.fetch_ohlcv(symbol, timeframe, since1,numb2)
    df2 = pd.DataFrame(ohlcv1, columns=['Date','Open','High','Low','Close','Volume'])
    df = pd.concat([df,df2])

  df['Date']=pd.to_datetime(df['Date']*1000000)
  print(name+"----------------------------")
  return df



def filt(a):
  # a['Volume'] = a['Volume']/1000
  # a['Volume'] = a[a['Volume']<100000]
  # a.pop('Unnamed: 0')
  a.pop('Volume')
  a.pop('Date')
  a = np.array(a)
  return a

def fetchdata(startdate,n,nn):
    # fetchData('.BLTC', 50, '1m', 300, 'BLTC.csv')
    # fetchData('.BUSDT', 50, '1m', 300, 'BUSDT.csv')
    ETHUSD = fetchData('ETH/USD', n, '5m', nn, 'ETHUSD5.csv', 5, startdate)
    BTCUSD = fetchData('BTC/USD', n, '5m', nn, 'BTCUSD5.csv', 5, startdate)

    # BLTC = filt('BLTC.csv')
    # BUSDT = filt('BUSDT.csv')
    ETHUSD = filt(ETHUSD)
    BTCUSD2 = BTCUSD.copy()
    BTCUSD1 = filt(BTCUSD)
    
    # print(BTCUSD2)
    # date = np.array(pd.to_datetime(BTCUSD2['Date'], format='%Y-%m-%d %H:%M:%S').dt.strftime('%H:%M'))
    # close = np.array(BTCUSD2['Close'])
    # high = np.array(BTCUSD2['High'])
    dataar = np.concatenate((BTCUSD1, ETHUSD), axis=1) 
    # print(date.shape, dataar.shape)
    print('get data................')
    return dataar#, date, close, high