# AI_trading

1. get to working directory : `cd ai_trading/tradingmodel`
2. to start training model : `python createmodel.py`
3. to start trading : `python trading.py`


**tradingmodel structure :**

_ components :_
- fetchdata : fetching data from bitmex with ccxt
- createdataset : cleaning and preprocessing data & creating time series dataset
- unifyclassessizes : uniffy classes sizes
- scaller : scalling
- createmodel : function for creating model layers

_ models :_ folder to save pretrained models

**createmodel.py file :**

- _algoritm of function **creating** work: _
fetchdata => createdataset(tradingmodel/createdataset.py) => traintestsplit => unnify classes sizes(tradingmodel/unifyclassessizes.py) => scalling(tradingmodel/scaller.py) => traintestsplit => createmodel(tradingmodel/createmodel.py) => fitmodel => testmodel => save model to tradingmodel/models

-_ logic of file _: call function creating twice - one for sell prediction `creating('sell', 'tradingmodel5msell')`, another for buy 
`creating('buy', 'tradingmodel5mbuy')`

**trading.py file :**

_ structure elements :_
- def makepredict : make prediction
- def fetchOrders : fetch orders
- def sowTickers : see what tickers are actual
- def createSellLimits : create sell limits
- def createBuyLimits : create buy limits
- def watching : take values for report

_ logic_: loop for trading operations. 
For more details look for comments inside loop

**checking.pyfile :**

- check if orders are changed, call from trading.py

        

