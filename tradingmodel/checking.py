import pandas as pd
import time
import tensorflow as tf
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
import time
import os
import ccxt
import requests
import json
import subprocess
import sys
from datetime import datetime, timedelta

def checking():
  # bitmex   = ccxt.bitmex({
  #   'apiKey': 'U2O9-r7CiBFOj8nC9XX8TWwu',
  #   'secret': 'jWrY9bV6PjSsFsF5JUv6HvrnbHB_71pSsuH8nBbY4Y9xHtPO',
  # })
  bitmex   = ccxt.bitmex({
      'apiKey': 'rHtA_bklMiwfmlRuJIkWthFt',
      'secret': 'i24DSLItVmj6kPAvEO2vu_dieK6Rn1VdLpUAaYhcrWb7Nlcj',
  })
  bitmex.urls['api'] = bitmex.urls['test']
  timeframe = '1m'

  def fetchOrders():
    openOrders = len(bitmex.fetch_open_orders())
    # print('LimitOrders: ', openOrders)
    isLessOrders = openOrders<2
    # print('isLessOrders : ',isLessOrders)
    return isLessOrders, openOrders
  def showTickers():
    cc = bitmex.fetchTickers()
    cc = pd.DataFrame(cc)
    cc1 = cc['BTC/USD'].bid
    cc2 = cc['BTC/USD'].ask
    print('CC',cc1, cc2)
    hight = round(cc2*1.002, 2)
    down = round(cc1*0.998, 2)
    print('H/D',hight, down)
    return hight, down
  # def createLimits():
  #   hight, down = showTickers()
  #   symbol = 'BTC/USD' 
  #   typee = 'StopLimit' 
  #   side = 'sell'
  #   amount = 25
  #   price = int(down) 
  #   params = {
  #     'stopPx': int(down), 
  #   }
  #   bitmex.create_order(symbol, typee, side, amount, price, params)
  #   bitmex.create_limit_sell_order('BTC/USD', 25, int(hight))
  #   print('Limits are created')

  import sched, time
  from datetime import datetime, timedelta

  # while True:
  isLessOrders, openOrders = fetchOrders()
  if isLessOrders:
    print('Cancel orders')
    bitmex.cancel_all_orders()
    ifend = True
    # time.sleep(60)
  else:
    ifend=False
  return ifend