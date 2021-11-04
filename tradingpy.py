import time
import numpy as np
import pandas as pd
from indicators import Bbands, average_true_range, buildMinutePrices, \
    appMinutePrices, buildBuckets, updateBuckets, getBucketSize, downloadTestData, runBacktest
from utilities import timestr

# TRADING RULES
QUANTPRE = {'BTCUSDT': 3, 'ETHUSDT': 3, 'BCHUSDT': 2, 'XRPUSDT': 1, 'EOSUSDT': 1, 'LTCUSDT': 3,
            # Precision for quantity
            'TRXUSDT': 0, 'ETCUSDT': 2, 'LINKUSDT': 2, 'XLMUSDT': 0, 'ADAUSDT': 0, 'XMRUSDT': 3,
            'DASHUSDT': 3, 'ZECUSDT': 3, 'XTZUSDT': 1, 'BNBUSDT': 2, 'ATOMUSDT': 2, 'ONTUSDT': 1,
            'IOTAUSDT': 1, 'BATUSDT': 1, 'VETUSDT': 0, 'NEOUSDT': 2, 'QTUMUSDT': 1, 'IOSTUSDT': 0,
            'OMGUSDT': 1, 'DOTUSDT': 1, 'SXPUSDT': 1}
PRICEPRE = {'BTCUSDT': 2, 'ETHUSDT': 2, 'BCHUSDT': 2, 'XRPUSDT': 4, 'EOSUSDT': 3, 'LTCUSDT': 2,  # Precision for price
            'TRXUSDT': 5, 'ETCUSDT': 3, 'LINKUSDT': 3, 'XLMUSDT': 5, 'ADAUSDT': 5, 'XMRUSDT': 2,
            'DASHUSDT': 2, 'ZECUSDT': 2, 'XTZUSDT': 3, 'BNBUSDT': 3, 'ATOMUSDT': 3, 'ONTUSDT': 4,
            'IOTAUSDT': 4, 'BATUSDT': 4, 'VETUSDT': 6, 'NEOUSDT': 3, 'QTUMUSDT': 3, 'IOSTUSDT': 6,
            'OMGUSDT': 4, 'DOTUSDT': 3, 'SXPUSDT': 4}

SIDE = {'BUY': 1.0, 'SELL': -1.0}

min_in_ms = int(60 * 1000)
sec_in_ms = 1000


class Portfolio:

    def __init__(self,
                 client,
                 tradeIns=[]):
        """
        Portfolio class
        """
        self.client = client
        self.tradeIns = tradeIns.copy()
        self.orderSize = 0
        self.equityDist = {'BUY': 0, 'SELL': 0}
        self.locks = {'BUY': [], 'SELL': []}

    def equity_distribution(self, longPct=0.5, shortPct=0.5, currency='USDT', orderPct=1):
        """
        Rerun number of buy/sell orders with currency equity

            longPct : percentage of equity assigned for buying

            shortPct : percentage of equity assigned for selling

            orderPct : percentage of equity for a single order
        """
        balance = self.client.balance()
        equity, available = 0, 0
        for b in balance:
            if b['asset'] == currency:
                equity, available = float(b['balance']), float(b['withdrawAvailable'])
                break

        long_equity = longPct * equity
        short_equity = shortPct * equity
        info = pd.DataFrame(self.client.position_info())
        short_info = info[info['positionAmt'].astype(float) < 0]
        long_info = info[info['positionAmt'].astype(float) > 0]
        short_position = abs(short_info['positionAmt'].astype(float) @ short_info['entryPrice'].astype(float))
        long_position = abs(long_info['positionAmt'].astype(float) @ long_info['entryPrice'].astype(float))

        self.orderSize = round(orderPct * equity, 2)
        long_order = int((long_equity - long_position) / self.orderSize)
        short_order = int((short_equity - short_position) / self.orderSize)
        self.equityDist = {'BUY': long_order, 'SELL': short_order}
        return long_order, short_order

    def position_locks(self, prelocks={'BUY': [], 'SELL': []}):
        """
        Check for open positions and return a tradeable instruments
        """
        info = self.client.position_info()
        self.locks = prelocks
        for pos in info:
            amt = float(pos['positionAmt'])
            if amt < 0 and not pos['symbol'] in self.locks['SELL']:
                self.locks['SELL'].append(pos['symbol'])
            elif amt > 0 and not pos['symbol'] in self.locks['BUY']:
                self.locks['BUY'].append(pos['symbol'])
        drop_out = set(self.locks['SELL']).intersection(self.locks['BUY'])
        for s in drop_out: self.tradeIns.remove(s)
        return self.tradeIns


class TradingModel:
    def __init__(self,
                 symbol: str,
                 testnet: bool,
                 modelType: str,
                 marketData,
                 pdObserve: int,
                 pdEstimate: int,
                 features: dict = None,
                 inputData=None,
                 orderSize=1.0,  # USDT
                 breath: float = 0.01 / 100,
                 minutePrices=None,
                 leftOver=None,
                 a=None,
                 buckets=None,
                 leftOverBuckets=None,
                 bucketSize=None,
                 backtest=None,
                 startTime=None):
        """
        Trading Model class
        :type backtest: True/False
        """
        self.symbol = symbol
        self.testnet = testnet
        self.modelType = modelType
        self.marketData = marketData
        self.pdObserve = pdObserve
        self.pdEstimate = pdEstimate
        self.inputData = inputData
        self.timeLimit = int(self.pdObserve * 10)
        self.orderSize = orderSize
        self.breath = breath
        self.signalLock = []
        self.long = 20
        self.short = 10
        self.minutePrices = minutePrices
        self.leftOver = leftOver
        self.a = a
        self.buckets = buckets
        self.leftOverBuckets = leftOverBuckets
        self.bucketSize = bucketSize
        self.backtest = backtest
        self.startTime = startTime
        self.testInputData = None
        self.testMinutePrices = None
        self.testLeftOver = None
        # accInfo = Client.account_info()
        # self.balance = accInfo["totalWalletBalance"]

    def add_signal_lock(self, slock=None):
        """
        Add a signal to lock positions i.e. abandon BUY/SELL the instrument
        """
        if (slock is not None) and (not slock in self.signalLock):
            self.signalLock.append(slock)

    def remove_signal_lock(self, slock=None):
        """
        Remove a signal from lock positions i.e. allows BUY/SELL the instrument
        """
        if (slock is not None) and (slock in self.signalLock):
            self.signalLock.remove(slock)

    def build_initial_input(self, period=180):
        """
        Download and store historical data
        """
        if self.modelType == 'cradle':
            if self.backtest:  # download last weeks tick data
                minsWorthTrades = 50000  # 525600 per year
                serverTime = self.marketData.server_time()['serverTime']
                startTime = serverTime - minsWorthTrades * 60 * 1000
                currentTime = startTime  # current Time = time we have collected data up to
                df = None
                print("scraping data: ")
                count = 1
                """while currentTime < serverTime:
                    # build the data to run the backtest
                    if currentTime == startTime:
                        df = self.marketData.aggregate_trades(startTime=currentTime, limit=1000)
                        df = pd.DataFrame(df)
                        currentTime = int(df['T'].iloc[-1])
                    else:
                        data = self.marketData.aggregate_trades(startTime=currentTime, limit=1000)
                        data = pd.DataFrame(data)
                        df = df.append(data)
                        currentTime = int(df['T'].iloc[-1])
                        print("Minutes Scraped: ", minsWorthTrades - (serverTime-currentTime)//60000)
                        #print("finished ", (currentTime - startTime)/60000)
                        if (minsWorthTrades - (serverTime-currentTime)//60000) > (count * 50000):
                            name = str(count) + "minutesdataBTC.csv"
                            df = pd.DataFrame(df)
                            df.to_csv(name)
                            print(df)
                            df = df[0:0]
                            count += 1
                            print(count)"""
                # df = pd.DataFrame(df)
                # print("DONE")
                # print(df)
                # df.to_csv('1yeardataBTC.csv')
                # downloadTestData(self.startTime, self.symbol, df)
                # read data in from excel
                buckets = [250]
                for i in buckets:
                    zScoreCalc = [0.5]
                    for j in zScoreCalc:
                        minutesInPrices = [1]
                        for k in minutesInPrices:
                            bucketTime = ['5m']
                            for z in bucketTime:
                                print("running test: ", i, j, k, z)
                                self.bucketSize = getBucketSize(self.marketData.candles_data(interval=z, limit=30),
                                                                numberOfBuckets=i)
                                finished, bucketSize = runBacktest(self.startTime, self.symbol, self.bucketSize, i, j,
                                                                   k, z)

                                print("finished", i, j, k, z)

                return
            if self.minutePrices is None:
                mins_worth_trades = 20
                t_server = self.marketData.server_time()['serverTime']
                t_start = t_server - mins_worth_trades * 60 * 1000
                df = self.marketData.aggregate_trades(startTime=t_start, limit=1000)
                df = pd.DataFrame(df)
                if len(df) == 1000:
                    startId = int(df['a'].iloc[-1]) + 1
                    data = self.marketData.aggregate_trades(fromId=startId, limit=1000)
                    data = pd.DataFrame(data)
                    df = df.append(data, ignore_index=True)
                    if len(data) == 1000:
                        startId = int(data['a'].iloc[-1]) + 1
                        data = self.marketData.aggregate_trades(fromId=startId, limit=1000)
                        data = pd.DataFrame(data)
                        df = df.append(data, ignore_index=True)
                        if len(data) == 1000:
                            startId = int(data['a'].iloc[-1]) + 1
                            data = self.marketData.aggregate_trades(fromId=startId, limit=1000)
                            data = pd.DataFrame(data)
                            df = df.append(data, ignore_index=True)
                            if len(data) == 1000:
                                startId = int(data['a'].iloc[-1]) + 1
                                data = self.marketData.aggregate_trades(fromId=startId, limit=1000)
                                data = pd.DataFrame(data)
                                df = df.append(data, ignore_index=True)
            if self.inputData is None:
                self.inputData = df
                print(df)
            else:
                df = df[df['T'] > self.inputData['T'].iloc[-1]]
                self.inputData = self.inputData.append(df, ignore_index=True)
            if self.minutePrices is None:
                self.a = 0.032
                self.minutePrices, self.leftOver = buildMinutePrices(self.inputData, self.a)
            if self.buckets is None:
                self.bucketSize = getBucketSize(self.marketData.candles_data(interval='1d', limit=30),
                                                numberOfBuckets=100)
                self.buckets, self.leftOverBuckets = buildBuckets(self.inputData, self.bucketSize)
        return self.inputData

    def get_last_signal(self, dataObserve=None):
        """
        Process the latest data for a potential signal
        """
        if self.modelType == 'cradle':
            _data = dataObserve[dataObserve['a'] > self.inputData['a'].iloc[-1]]
            self.inputData = self.inputData.append(_data, ignore_index=True)
            self.inputData = self.inputData.iloc[1:]
            self.inputData, self.minutePrices, self.leftOver, updated = appMinutePrices(self.inputData,
                                                                                        self.minutePrices,
                                                                                        self.leftOver, self.a)
            self.buckets, self.leftOverBuckets = updateBuckets(self.inputData, self.buckets,
                                                               self.leftOverBuckets, self.bucketSize)

            if updated:  # vpin
                if len(self.minutePrices) > 10:
                    self.minutePrices = self.minutePrices.iloc[1:]
                print("updated")
            close = _data['_c'].iloc[-1]
            open = _data['_o'].iloc[-1]
            high = _data['_h'].iloc[-1]
            low = _data['_l'].iloc[-1]
            # balance = self.balance
            balance = 5  # change account balance
            maxLoss = 0.00175 * balance
            slippage = 0.0002
            priceDiff = 0
            symbol = str(self.symbol.upper())
            if PRICEPRE[symbol] == 1:
                priceDiff = 0.1
            elif PRICEPRE[symbol] == 2:
                priceDiff = 0.01
            elif PRICEPRE[symbol] == 3:
                priceDiff = 0.001
            elif PRICEPRE[symbol] == 4:
                priceDiff = 0.0001
            elif PRICEPRE[symbol] == 5:
                priceDiff = 0.00001
            elif PRICEPRE[symbol] == 6:
                priceDiff = 0.000001
            elif PRICEPRE[symbol] == 7:
                priceDiff = 0.0000001
            commission = 0.00028
            if shortEma > longEma and trend == "uptrend" and high < (shortEma * 1.0005) and low > (
                    longEma * 0.9995) and not 'BUY' in self.signalLock and close > open:
                posLow = getHLPrice(_data, "UP", 2)
                posHigh = high + priceDiff
                posLow = posLow - priceDiff
                stopPrice = posHigh * (1 + slippage)
                stopPriceSL = posLow * (1 - slippage)
                size = (maxLoss / (stopPrice - stopPriceSL)) * stopPrice
                if size > 2 * balance:
                    size = 2 * balance
                takeProfit = (stopPrice - stopPriceSL)
                return {'side': 'BUY', 'positionSide': 'LONG', '_t': _data['_t'].iloc[-1],
                        '_p': posHigh, 'sl': posLow, 'tp': takeProfit, 'size': size, 'type': 'STOP', 'stop': stopPrice,
                        'stopSL': stopPriceSL}

            elif shortEma < longEma and trend == "downtrend" and high < (longEma * 1.0005) and low > (
                    shortEma * 0.9995) and not 'SELL' in self.signalLock and close < open:
                posHigh = getHLPrice(_data, "DOWN", 2)
                posLow = low - priceDiff
                posHigh = posHigh + priceDiff
                stopPrice = posLow * (1 - slippage)
                stopPriceSL = posHigh * (1 + slippage)
                size = (maxLoss / (stopPriceSL - stopPrice)) * stopPrice
                if size > 2 * balance:
                    size = 2 * balance
                takeProfit = (stopPriceSL - stopPrice)
                return {'side': 'SELL', 'positionSide': 'SHORT', '_t': _data['_t'].iloc[-1],
                        '_p': posLow, 'sl': posHigh, 'tp': takeProfit, 'size': size, 'type': 'STOP', 'stop': stopPrice,
                        'stopSL': stopPriceSL}
        return None


class Signal:
    def __init__(self,
                 symbol: str,
                 side: str,
                 size: float,
                 orderType: str,
                 positionSide: str = 'BOTH',
                 price: float = None,
                 startTime: int = time.time() * 1000,
                 expTime: float = (time.time() + 60) * 1000,
                 stopLoss: float = None,
                 takeProfit: float = None,
                 timeLimit: int = None,  # minutes
                 timeInForce: float = None,
                 stopPrice: float = None,
                 stopPriceSL: float = None):
        """

        Signal class to monitor price movements

        To change currency pair     -> symbol = 'ethusdt'

        To change side              -> side = 'BUY'/'SELL'

        To change order size        -> size = float (dollar amount)

        To change order type        -> orderType = 'MARKET'/'LIMIT'

        To change price             -> price = float (required for 'LIMIT' order type)

        stopLoss, takeProfit -- dollar amount

        To change time in force     -> timeInForce =  'GTC'/'IOC'/'FOK' (reuired for 'LIMIT' order type)

        """
        self.stopPrice = stopPrice
        self.stopPriceSl = stopPriceSL
        self.symbol = symbol
        self.side = side  # BUY, SELL
        self.positionSide = positionSide  # LONG, SHORT
        self.orderType = orderType  # LIMIT, MARKET, STOP, TAKE_PROFIT
        # predefined vars
        self.price = float(price)
        if size < self.price * 10 ** (-QUANTPRE[symbol]):
            size = self.price * 10 ** (-QUANTPRE[symbol]) * 1.01
        self.size = float(size)  # USDT
        self.quantity = round(self.size / self.price, QUANTPRE[self.symbol])
        self.startTime = int(startTime)
        self.expTime = expTime
        self.stopLoss = round(stopLoss, PRICEPRE[symbol])
        # 3 exit barriers
        # if stopLoss is not None:
        #   self.stopLoss = round(float(stopLoss), 4)
        # else:
        #   self.stopLoss = None
        self.takeProfit = round(takeProfit, PRICEPRE[symbol])
        # if takeProfit is not None:
        #   self.takeProfit = round(float(takeProfit), 4)
        # else:
        #   self.takeProfit = None
        if timeLimit is not None:
            self.timeLimit = int(timeLimit * sec_in_ms)  # miliseconds
        else:
            self.timeLimit = None

        self.timeInForce = timeInForce
        self.status = 'WAITING'  # 'ORDERED' #'ACTIVE' #'CNT_ORDERED' #'CLOSED' # 'EXPIRED'
        self.limitPrice, self.orderTime = None, None
        self.excPrice, self.excTime = None, None
        self.cntlimitPrice, self.cntTime, self.cntType = None, None, None
        self.clsPrice, self.clsTime = None, None
        self.orderId = None
        self.cntorderId = None
        self.pricePath = []
        self.exitSign = None
        self.newStopLoss = None
        self.trailorderId = None
        self.trailRate = None
        self.orderSLId = None
        self.orderTPId = None
        self.trigger = None

    '''
    Function to check and set STATUS of the signals : 
        - WAITING
        - ORDERED
        - ACTIVE
        - CNT_ORDERED
        - CLOSED
        - EXPIRED
    '''

    def is_waiting(self):
        return bool(self.status == 'WAITING')

    def set_waiting(self):
        self.status = 'WAITING'

    def is_ordered(self):
        return bool(self.status == 'ORDERED')

    def set_ordered(self, orderId, orderTime=None, limitPrice=None, stopPrice=None, orderSLId=None):
        self.status = 'ORDERED'
        self.orderId = int(orderId)
        self.orderTime, self.limitPrice, self.stopPrice = orderTime, limitPrice, stopPrice
        self.orderSLId = int(orderSLId)

    def is_active(self):
        return bool(self.status == 'ACTIVE')

    def set_active(self, excTime=time.time() * 1000, excPrice=None, excQty: float = None, orderIdTP: float = None,
                   orderIdSL: float = None):
        self.excPrice = float(excPrice)
        self.excTime = int(excTime)
        self.quantity = round(float(excQty), QUANTPRE[self.symbol])
        self.orderTPId = orderIdTP
        self.orderSLId = orderIdSL
        if self.side == "BUY":
            self.newStopLoss = self.stopLoss - self.excPrice
        elif self.side == "SELL":
            self.newStopLoss = self.excPrice - self.stopLoss
        self.status = 'ACTIVE'

    def is_cnt_ordered(self):
        return bool(self.status == 'CNT_ORDERED')

    def set_cnt_ordered(self, cntorderId, cntType=None, cntTime=None, cntlimitPrice=None, trailorderId=None,
                        trailRate=None, trigger=None):
        if trailorderId is not None:
            self.trailorderId = int(trailorderId)
            self.trailRate = trailRate
        self.status = 'CNT_ORDERED'
        self.cntorderId = int(cntorderId)
        self.trigger = trigger
        self.cntType, self.cntTime, self.cntlimitPrice = cntType, cntTime, cntlimitPrice

    def is_closed(self):
        return bool(self.status == 'CLOSED')

    def set_closed(self, clsTime=time.time() * 1000, clsPrice=None):
        self.clsTime = int(clsTime)
        if clsPrice is not None:
            self.clsPrice = float(clsPrice)
        else:
            self.clsPrice = None
        self.status = 'CLOSED'

    def is_expired(self):
        return bool(self.status == 'EXPIRED')

    def set_expired(self):
        self.status = 'EXPIRED'

    def get_quantity(self):
        """
        Return quantity
        """
        return self.quantity

    def counter_order(self):
        """
        Return counter (close) order with same size but opposite side
        """
        if self.side == 'BUY':
            side = 'SELL'
        else:
            side = 'BUY'
        if self.positionSide == 'LONG':
            posSide = 'SHORT'
        elif self.positionSide == 'SHORT':
            posSide = 'LONG'
        else:
            posSide = 'BOTH'
        counter = {'side': side, 'positionSide': posSide, 'type': self.orderType,
                   'amt': self.get_quantity(), 'TIF': self.timeInForce}
        return counter

    def path_update(self, lastPrice, lastTime):
        """
        Update last traded prices to pricePath
        """
        self.pricePath.append({'timestamp': int(lastTime), 'price': float(lastPrice)})

    def get_price_path(self):
        """
        Return price movements since the entry
        """
        return pd.DataFrame(self.pricePath)

    def exit_triggers(self, lastTime=None, lastPrice=None, retrace=False):
        """
        Return a exit signal upon 3 barrier triggers
        """
        foundSL = None
        if not self.is_active() or len(self.pricePath) <= 1:
            return None, None
        else:
            exit_sign = False
            if lastTime is None and lastPrice is None:
                _t, _p = self.pricePath[-1]['timestamp'], self.pricePath[-1]['price']  # _t = last time, _p = last price
            pos = SIDE[self.side] * (
                    _p - self.excPrice)  # buy = 1, sell = -1, pos = neg when out of the money, price change
            # if self.takeProfit is not None and pos > self.takeProfit:  # take profit is positive price change
            #   exit_sign = 'takeProfit'
            # if self.newStopLoss is not None:  # new stop loss = price change
            #   if retrace:
            #      prices = pd.DataFrame(self.pricePath)
            #    prices['pos'] = SIDE[self.side] * (
            #           prices['price'] - self.excPrice)  # prices['pos'] = neg when out of the money
            #  if self.side == "BUY":
            #     loss_idx = prices.idxmin(axis=0)['price']  # index of min price
            #    minPrice = prices.loc[loss_idx]['price']  # min price since entry
            #   if (minPrice < self.stopLoss) and (pos < self.newStopLoss):
            #      foundSL = True
            #        elif self.side == "SELL":
            #           loss_idx = prices.idxmax(axis=0)['price']  # index of max price
            #          maxPrice = prices.loc[loss_idx]['price']  # max price since entry
            #         if (maxPrice > self.stopLoss) and (pos < self.newStopLoss):
            #            foundSL = True
            # if foundSL:
            #   exit_sign = 'stopLoss'
            if self.timeLimit is not None and _t - self.excTime >= self.timeLimit and pos > 0:
                exit_sign = 'timeLimit'
            self.exitSign = exit_sign
            return exit_sign, pos

    def __str__(self):
        """
        Print out information of the signal
        """
        s = 'Signal info: ' + self.symbol
        gen_ = ' status:' + str(self.status) + ' side:' + str(self.side) + ' type:' + str(
            self.orderType) + ' quantity:' + str(self.get_quantity())
        if self.is_waiting() or self.is_expired():
            id_ = ' Id:None '
            price_ = ' price:' + str(self.price) + ' time:' + timestr(self.startTime, end='s')
        elif self.is_ordered():
            id_ = ' Id:' + str(self.orderId)
            if self.orderType == 'LIMIT':
                price_ = ' price:' + str(self.limitPrice) + ' TIF:' + str(self.timeInForce) + ' time:' + timestr(
                    self.startTime, end='s')
            else:
                price_ = ' type:' + str(self.orderType) + ' time:' + timestr(self.orderTime, end='s')
        elif self.is_active():
            id_ = ' Id:' + str(self.orderId)
            if self.orderType == 'LIMIT':
                price_ = ' price:' + str(self.excPrice) + ' TIF:' + str(self.timeInForce) + ' time:' + timestr(
                    self.excTime, end='s')
            else:
                price_ = ' price:' + str(self.excPrice) + ' time:' + timestr(self.excTime, end='s')
        elif self.is_cnt_ordered():
            gen_ = ' status:' + str(self.status) + ' side:' + str(self.counter_order()['side']) + ' type:' + str(
                self.cntType) + ' quantity:' + str(self.get_quantity())
            id_ = ' Id:' + str(self.cntorderId)
            if self.cntType == 'LIMIT':
                price_ = ' price:' + str(self.cntlimitPrice) + ' TIF:' + str(self.timeInForce) + ' time:' + timestr(
                    self.cntTime, end='s')
            else:
                price_ = ' type:' + str(self.cntType) + ' time:' + timestr(self.cntTime, end='s')
        elif self.is_closed():
            gen_ = ' status:' + str(self.status) + ' side:' + str(self.counter_order()['side']) + ' type:' + str(
                self.cntType) + ' quantity:' + str(self.get_quantity())
            id_ = ' Id: ' + str(self.cntorderId)
            price_ = ' price:' + str(self.clsPrice) + ' time:' + timestr(self.clsTime, end='s')
        if self.stopLoss is None:
            sl_ = 'None'
        else:
            sl_ = str(self.stopLoss)
        if self.takeProfit is None:
            tp_ = 'None'
        else:
            tp_ = str(self.takeProfit)
        if self.timeLimit is None:
            tl_ = 'None'
        else:
            tl_ = str(int(self.timeLimit / sec_in_ms))
        exits_ = ' exits:[' + sl_ + ', ' + tp_ + ', ' + tl_ + ']'
        s += id_ + gen_ + price_ + exits_
        return s


def klns_to_df(market_data, feats):
    """
    Return a pd.DataFrame from candles data received from the exchange
    """
    fts = list(str(f) for f in feats)
    df_ = pd.DataFrame(market_data,
                       columns=['_t', '_o', '_h', '_l', '_c', '_v', 'close_time', 'quote_av', 'trades', 'tb_base_av',
                                'tb_quote_av', 'ignore'])
    df_[['_o', '_h', '_l', '_c', '_v']] = df_[['_o', '_h', '_l', '_c', '_v']].astype(float)
    return df_[fts]


def aggTrades_to_df(market_data, feats):
    """
    :return a pd.DataFrame from trade data received from exchange
    """


"""    fts = list(str(f) for f in feats)
    df_ = pd.DataFrame(market_data,
                       columns=[''])
    "a": 26129, // Aggregate
    tradeId
    "p": "0.01633102", // Price
    "q": "4.70443515", // Quantity
    "f": 27781, // First
    tradeId
    "l": 27781, // Last
    tradeId
    "T": 1498793709153, // Timestamp
    "m": true,"""
