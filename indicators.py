import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pandas import DataFrame, Series
from pandas.core.generic import NDFrame
from pandas.io.parsers import TextFileReader
from trendIndicator import calcTrend, appendTrend

def EMA(df, short, long):
    """
    Calculate the exponential moving average
    :param df: data frame
    :param short: length of short ema
    :param long: length of long ema
    :return: ema short and long
    """
    emaSHORT = {}
    smaShort = df['_c'].rolling(short).mean()
    modPriceShort = df['_c'].copy()
    modPriceShort.iloc[0:short] = smaShort[0:short]
    emaSHORT['emaShort'] = modPriceShort.ewm(span=short, adjust=False).mean()
    emaShort = emaSHORT['emaShort'].iloc[-1]

    emaLONG = {}
    smaLong = df['_c'].rolling(long).mean()
    modPriceLong = df['_c'].copy()
    modPriceLong.iloc[0:long] = smaLong[0:long]
    emaLONG['emaLong'] = modPriceLong.ewm(span=long, adjust=False).mean()
    emaLong = emaLONG['emaLong'].iloc[-1]

    return emaShort, emaLong


def buildTrendDirection(minutePrices, a):
    ehlers = {}
    dfCopy = minutePrices.copy()
    for i in range(len(minutePrices)):
        if i % 10000 == 0:
            print("count: {}, length: {}".format(i, len(ehlers)))
        if i < 2:
            ehlers.update({minutePrices['t'][i]: {'it': 'N/A', 'trend': 'notrend'}})
        elif i == 2:
            diff = len(minutePrices) - 1 - i
            count = len(minutePrices) - 1 - diff
            it = (a - ((a * a) / 4.0)) * dfCopy['p'].shift(count) + (
                    0.5 * a * a * dfCopy['p'].shift(count - 1)) - (
                         (a - (0.75 * a * a)) * dfCopy['p'].shift(count - 2)) + (
                         2 * (1 - a) * ((dfCopy['p'].shift(count) + 2 *
                                         dfCopy['p'].shift(count - 1) + dfCopy[
                                             'p'].shift(count - 2)) / 4.0)) - \
                 ((1 - a) * (1 - a) * (
                         (dfCopy['p'].shift(count) + 2 * dfCopy['p'].shift(count - 1) + dfCopy['p'].shift(
                             count - 2)) / 4.0))
            ehlers.update({minutePrices['t'][i]: {'it': float(it.iloc[count]), 'trend': 'notrend'}})
        elif i == 3:
            diff = len(minutePrices) - 1 - i
            count = len(minutePrices) - 1 - diff
            it = (a - ((a * a) / 4.0)) * dfCopy['p'].shift(count) + (
                    0.5 * a * a * dfCopy['p'].shift(count - 1)) - (
                         (a - (0.75 * a * a)) * dfCopy['p'].shift(count - 2)) + (
                         2 * (1 - a) * ehlers[minutePrices['t'][i - 1]]['it']) - ((1 - a) * (1 - a) * (
                    (dfCopy['p'].shift(count) + 2 * dfCopy['p'].shift(count - 1) + dfCopy['p'].shift(
                        count - 2)) / 4.0))
            ehlers.update({minutePrices['t'][i]: {'it': float(it.iloc[count]), 'trend': 'notrend'}})
        elif i > 3:
            diff = len(minutePrices) - 1 - i
            count = len(minutePrices) - 1 - diff
            it = (a - ((a * a) / 4.0)) * dfCopy['p'].shift(count) + (
                    0.5 * a * a * dfCopy['p'].shift(count - 1)) - (
                         (a - (0.75 * a * a)) * dfCopy['p'].shift(count - 2)) + (
                         2 * (1 - a) * ehlers[minutePrices['t'][i - 1]]['it']) - (
                         (1 - a) * (1 - a) * ehlers[minutePrices['t'][i - 2]]['it'])
            lag = 2.0 * float(it.iloc[count]) - float(ehlers[minutePrices['t'][i - 2]]['it'])
            if lag > float(it.iloc[count]):
                ehlers.update({minutePrices['t'][i]: {'it': float(it.iloc[count]), 'trend': 'uptrend'}})
            elif lag < float(it.iloc[count]):
                ehlers.update({minutePrices['t'][i]: {'it': float(it.iloc[count]), 'trend': 'downtrend'}})
            else:
                ehlers.update({minutePrices['t'][i]: {'it': float(it.iloc[count]), 'trend': 'notrend'}})
    print("DONE")
    ehlers = pd.DataFrame(ehlers)
    ehlers = ehlers.transpose()
    minutePrices.set_index('t', inplace=True)
    ehlers = minutePrices.join(ehlers)
    print(ehlers[0:])
    print("FINISHED")
    return ehlers


def getTrendDirection(df, minutePrices, a, currentTime):
    """
    append leftOver with it and trend
    :param currentTime:
    :param minutePrices: minute prices df
    :param df: {time: {v:float, p:float}}
    :param a: alpha used for calculation
    :return: trend direction
    """
    dfCopy = minutePrices.copy()
    leftOver = df
    volume = leftOver[currentTime]['v']
    price = leftOver[currentTime]['p']
    price2 = dfCopy['p'].iloc[-1]
    price3 = dfCopy['p'].iloc[-2]
    it = (a - ((a * a) / 4.0)) * price + (
            0.5 * a * a * price2) - (
                 (a - (0.75 * a * a)) * price3) + (
                 2 * (1 - a) * dfCopy['it'].iloc[-1]) - (
                 (1 - a) * (1 - a) * dfCopy['it'].iloc[-2])
    lag = 2.0 * float(it) - float(dfCopy['it'].iloc[-2])
    if lag > float(it):
        leftOver = {currentTime: {'v': volume, 'p': price, 'it': float(it), 'trend': 'uptrend'}}
    elif lag < float(it):
        leftOver = {currentTime: {'v': volume, 'p': price, 'it': float(it), 'trend': 'downtrend'}}
    else:
        leftOver = {currentTime: {'v': volume, 'p': price, 'it': float(it), 'trend': 'notrend'}}
    # print(leftOver)
    return leftOver


def getHLPrice(df, direction, lookBack):
    dfCopy = df.copy()
    price = 0
    if direction == "UP":
        count = lookBack  # number of previous candles checked for low
        price = dfCopy['_l'].iloc[-count]
        while count > 1:
            num = dfCopy['_l'].iloc[-count + 1]
            if num < price:
                price = num
            count -= 1
    else:
        count = lookBack  # number of previous candles checked for high
        price = dfCopy['_h'].iloc[-count]
        while count > 1:
            num = dfCopy['_h'].iloc[-count + 1]
            if num > price:
                price = num
            count -= 1

    return price


def buildMinutePrices(df, a, mins):
    count = 0
    volume = 0
    priceVol = 0
    time = 0
    prices = []
    leftOver = {}
    if count == 0:
        time = df['T'][count]
    new_T = df['T'].astype(int)
    for i in range(len(df)):
        if count % 100000 == 0:
            print("count: {}, minutes: {}".format(count, len(prices)))
        if new_T[count] <= time + 60000 * mins:  # int(df['T'][count])
            volume += float(df['q'][count])
            priceVol += float(df['q'][count]) * float(df['p'][count])
            count += 1
        else:
            price = priceVol / volume
            prices.append({'t': int(time), 'v': float(volume), 'p': float(price)})
            time = df['T'][count]
            volume = float(df['q'][count])
            priceVol = float(df['q'][count]) * float(df['p'][count])
            count += 1
        if count == len(df):
            price = priceVol / volume
            leftOver = {int(time): {'v': float(volume), 'p': float(price)}}
    prices = pd.DataFrame(prices)
    print("done")
    prices = calcTrend(prices, a)
    #prices = buildTrendDirection(prices, a)
    # time from start of minute
    return prices, leftOver


def appMinutePrices(inputData, pricesData, leftOver, a, mins):  # input, minuteprices, leftover
    inputData = inputData.copy()
    pricesData = pricesData.copy()
    currentTime = list(leftOver.keys())
    currentTime = currentTime[0]
    lastTime = int(inputData['T'].iloc[-1])
    if lastTime > currentTime + 60000 * mins:  # new minute
        leftOver = getTrendDirection(leftOver, pricesData, a, currentTime)
        newMinute = pd.DataFrame(leftOver)
        newMinute = newMinute.transpose()
        pricesData = pricesData.append(newMinute)
        update = True
        leftOver = {int(inputData['T'].iloc[-1]): {'v': float(inputData['q'].iloc[-1]),
                                                   'p': float(inputData['p'].iloc[-1])}}
        new = {int(inputData['T'].iloc[-1]): {'v': float(inputData['q'].iloc[-1]),
                                              'p': float(inputData['p'].iloc[-1])}}
    else:  # same minute
        volume = leftOver[currentTime]['v']
        priceVol = leftOver[currentTime]['v'] * leftOver[currentTime]['p']
        volume += float(inputData['q'].iloc[-1])
        priceVol += float(inputData['q'].iloc[-1]) * float(inputData['p'].iloc[-1])
        price = priceVol / volume
        update = False
        leftOver = {int(currentTime): {'v': float(volume), 'p': float(price)}}
    return inputData, pricesData, leftOver, update


def buildBuckets(df, bucketSize):
    count = 0
    volume = 0
    priceVol = 0
    price = 0
    newVolume = 0
    plusMinus = 0
    buckets = []
    leftOver = {}
    if count == 0:
        newVolume = float(df['q'][count])
        price = float(df['p'][count])
    for i in range(len(df)):
        if newVolume <= bucketSize:
            newVolume += float(df['q'][count])
            newPrice = float(df['p'][count])
            if newPrice > price:
                plusMinus += float(df['q'][count])
            elif newPrice < price:
                plusMinus -= float(df['q'][count])
            else:
                plusMinus += 0
            count += 1
            price = newPrice
        else:
            buckets.append({'+-': float(plusMinus), 'closed': True})
            newVolume = float(df['q'][count])
            price = float(df['p'][count])
            plusMinus = 0
            count += 1
        if count == len(df):
            leftOver = {'+-': float(plusMinus), 'v': float(newVolume), 'closed': False}
    buckets = pd.DataFrame(buckets)
    # time from start of minute
    return buckets, leftOver


def updateBuckets(inputData, buckets, leftOver, bucketSize):
    inputData = inputData.copy()
    buckets = buckets.copy()
    test = []
    volume = 0
    newVolume = leftOver['v']
    plusMinus = leftOver['+-']
    lastTime = int(inputData['T'].iloc[-1])
    if newVolume <= bucketSize:  # same bucket
        newVolume += float(inputData['q'].iloc[-1])
        newPrice = float(inputData['p'].iloc[-1])
        price = float(inputData['p'].iloc[-2])
        if newPrice > price:
            plusMinus += float(inputData['q'].iloc[-1])
        elif newPrice < price:
            plusMinus -= float(inputData['q'].iloc[-1])
        else:
            plusMinus += 0
        leftOver = {'+-': float(plusMinus), 'v': float(newVolume), 'closed': False}
    else:  # new bucket
        newBucket = pd.DataFrame({'+-': float(plusMinus), 'closed': True}, index=[0])
        buckets = buckets.append(newBucket, ignore_index=True)
        leftOver = {'+-': 0, 'v': 0, 'closed': False}
    return buckets, leftOver


def getBucketSize(data, numberOfBuckets=None):
    data.pop()
    count = 0
    total = 0
    for i in data:
        total += float(i[5])
        count += 1
    average = total / count
    bucketSize = average / numberOfBuckets
    return bucketSize


def downloadTestData(startTime, symbol, df):
    st = str(startTime)
    st = st[-6:]
    name = 'backtestdata/' + str(symbol) + st + '.xlsx'
    name = str(name)
    writer = pd.ExcelWriter(name, engine='openpyxl')
    df.to_excel(writer, sheet_name=symbol, index=False)
    writer.save()
    return


def readTestData(startTime, symbol, minutes, reRun=False):
    st = str(startTime)
    st = st[-6:]
    name = 'backtestdata/' + str(symbol) + st + '.xlsx'
    name = str(name)
    data = pd.read_excel(name, sheet_name=symbol,
                         dtype={'a': int, 'p': float, 'q': float, 'f': int, 'l': int, 'T': int, 'm': bool})
    inputData = []
    firstTime = data['T'].iloc[0]
    lastTime = 0
    length = len(data)
    for i in range(len(data)):
        if data['T'].iloc[i] < firstTime + (minutes * 60000):
            inputData.append(
                {'a': data['a'].iloc[i], 'p': data['p'].iloc[i], 'q': data['q'].iloc[i], 'f': data['f'].iloc[i],
                 'l': data['l'].iloc[i], 'T': data['T'].iloc[i], 'm': data['m'].iloc[i]})
            lastTime = i
    inputData = pd.DataFrame(inputData)
    # print(data)
    # print(inputData)
    return inputData, lastTime, length, data


def getLastTick(i, data):
    dataObserve = {'a': data['a'].iloc[i], 'p': data['p'].iloc[i], 'q': data['q'].iloc[i], 'f': data['f'].iloc[i],
                   'l': data['l'].iloc[i], 'T': data['T'].iloc[i], 'm': data['m'].iloc[i]}
    return dataObserve


def variance(data, ddof=0):
    n = len(data)
    if n < 3:
        return None
    else:
        mean = sum(data) / n
        return sum((x - mean) ** 2 for x in data) / (n - ddof)


def zScore(sd, x, mean=0):
    z = float((x - mean) / sd)
    return z


def calcVpin(leftOverBuckets, buckets, bucketSize, calc):
    data = []
    for i in range(len(buckets)):
        data.append(buckets['+-'].iloc[i])
    var = variance(data, ddof=1)
    if var is None:
        z = 0
    else:
        std_dev = math.sqrt(var)
        if leftOverBuckets['v'] > calc * bucketSize:
            mult = bucketSize / leftOverBuckets['v']
            value = leftOverBuckets['+-'] * mult
            z = zScore(std_dev, value, mean=0)
        else:
            value = data[-1]
            z = zScore(std_dev, value, mean=0)
    return z


def runBacktest(startTime, symbol, bucketSize, noBuckets, zCalc, minsInPrice, bucketTime):
    #testInputData, lastTime, length, data = readTestData(startTime, symbol, 7)
    for i in range(1):
        file_name = str(i + 1) + "minutesdataBTC.csv"
        print(file_name)
        if i == 0:
            testInputData = pd.read_csv(file_name)
            print(testInputData)
        if i > 0:
            testInputData = testInputData.append(pd.read_csv(file_name))
    testInputData = testInputData.reset_index(drop=True)
    print(testInputData)
    #testInputData.info()
    a = 0.01
    testMinutePrices, testLeftOver = buildMinutePrices(testInputData, a, minsInPrice)
    #buckets, leftOverBuckets = buildBuckets(testInputData, bucketSize)
    order = "Waiting"
    orderZ = 0
    direction = None
    orders = []
    orderPrice = 0
    orderSize = 1
    # read rest of the tick data and run algo
    """while lastTime < length - 1:
        lastTime += 1
        # read through rest of the tick data and return each line separately, return each line
        _data = getLastTick(lastTime, data)
        testInputData = testInputData.append(_data, ignore_index=True)
        testInputData = testInputData.iloc[1:]
        testInputData, testMinutePrices, testLeftOver, updated = appMinutePrices(testInputData,
                                                                                 testMinutePrices,
                                                                                 testLeftOver, a, minsInPrice)
        buckets, leftOverBuckets = updateBuckets(testInputData, buckets,
                                                 leftOverBuckets, bucketSize)
        # print(testMinutePrices)
        # if updated:  # vpin to maintain a small size
        #   if len(testMinutePrices) > 10:
        #      testMinutePrices = testMinutePrices.iloc[1:]
        if len(buckets) > 100:
            buckets = buckets.iloc[1:]
        z_score = calcVpin(leftOverBuckets, buckets, bucketSize, zCalc)
        trend = testMinutePrices['trend'].iloc[-1]
        if order == "Waiting":  # no current order
            # print(trend, z_score)
            if trend == 'uptrend' and z_score > 1.5:
                # print("Buy Signal")
                # print(trend, z_score)
                orderPrice = float(testInputData['p'].iloc[-1])
                orderSize = 1
                orderZ = z_score
                order = "Active"
                direction = "Buy"
                # orders = orders.append({'dir': direction, 'p': orderPrice, 's': orderSize, 'z': orderZ})
            elif trend == 'downtrend' and z_score < -1.5:
                # print("Sell Signal")
                # print(trend, z_score)
                orderPrice = float(testInputData['p'].iloc[-1])
                orderSize = 1
                orderZ = z_score
                order = "Active"
                direction = "Sell"
                # orders = orders.append({'dir': direction, 'p': orderPrice, 's': orderSize, 'z': orderZ})
        if order == "Active":  # current order placed
            # print(trend, z_score)
            if direction == "Buy":
                if trend == "downtrend" or z_score < 0:
                    # print("Close Buy Signal")
                    # print(trend, z_score)
                    closePrice = float(testInputData['p'].iloc[-1])
                    order = "Waiting"
                    orders.append({'dir': direction, 'pO': orderPrice, 'pC': closePrice, 's': orderSize, 'zO': orderZ,
                                   'zC': z_score})
                    # print(orders)
                    direction = None
                    orderPrice = None
                    orderSize = None
                    orderZ = None
            if direction == "Sell":
                if trend == "uptrend" or z_score > 0:
                    # print("Close Sell Signal")
                    # print(trend, z_score)
                    closePrice = float(testInputData['p'].iloc[-1])
                    order = "Waiting"
                    orders.append({'dir': direction, 'pO': orderPrice, 'pC': closePrice, 's': orderSize, 'zO': orderZ,
                                   'zC': z_score})
                    # print(orders)
                    direction = None
                    orderPrice = None
                    orderSize = None
                    orderZ = None

    orders = pd.DataFrame(orders)
    st = str(startTime)
    st = st[-6:]
    zCalc = str(zCalc)
    reg = re.compile(r'[\\/.:*?"<>|\r\n]+')
    valid_name = reg.findall(zCalc)
    if valid_name:
        for nv in valid_name:
            zCalc = zCalc.replace(nv, "_")
    name = 'backtestdata/' + str(symbol) + st + '.xlsx'
    name = str(name)
    sheetName = "b" + str(noBuckets) + "z" + str(zCalc) + "p" + str(minsInPrice) + "bT" + bucketTime
    writer = pd.ExcelWriter(name, engine='openpyxl', mode='a')
    # orders.to_excel(writer, sheet_name=sheetName, index=False)
    writer.save()
    # print(testMinutePrices, testLeftOver, buckets, leftOverBuckets, bucketSize)
    bucketSize = bucketSize"""
    testMinutePrices.index.name = "t"
    testMinutePrices = testMinutePrices.reset_index()
    from datetime import datetime, timedelta
    testMinutePrices['t'] = pd.to_datetime(testMinutePrices['t'], unit='ms')
    print(testMinutePrices)
    # epoch = datetime(1601, 1, 1)
    # cookie_microseconds_since_epoch = testMinutePrices['t'][2]
    # cookie_datetime = epoch + timedelta(microseconds=cookie_microseconds_since_epoch)
    # str(cookie_datetime)
    testMinutePrices.to_csv('file_name.csv')
    fig = px.line(testMinutePrices[200:1400], x="t", y="p")
    for i, d in enumerate(fig.data):
        for j, y in enumerate(d.y):
                if j % 1000 == 0:
                    print(j, "DONE")
                if j == 0:
                    fig.add_traces(go.Scatter(x=[fig.data[i]['x'][j]],
                                              y=[fig.data[i]['y'][j]],
                                              mode='markers',
                                              marker=dict(color='yellow')))
                else:
                    if testMinutePrices['trend'][j] == "notrend":
                        fig.add_traces(go.Scatter(x=[fig.data[i]['x'][j - 1], fig.data[i]['x'][j]],
                                                  y=[fig.data[i]['y'][j - 1], fig.data[i]['y'][j]],
                                                  mode='lines',
                                                  # marker = dict(color='yellow'),
                                                  line=dict(width=5, color='yellow')
                                                  ))
                    elif testMinutePrices['trend'][j] == "downtrend":
                        fig.add_traces(go.Scatter(x=[fig.data[i]['x'][j - 1], fig.data[i]['x'][j]],
                                                  y=[fig.data[i]['y'][j - 1], fig.data[i]['y'][j]],
                                                  mode='lines',
                                                  # marker = dict(color='yellow'),
                                                  line=dict(width=5, color='red')
                                                  ))
                    elif testMinutePrices['trend'][j] == "uptrend":
                        fig.add_traces(go.Scatter(x=[fig.data[i]['x'][j - 1], fig.data[i]['x'][j]],
                                                  y=[fig.data[i]['y'][j - 1], fig.data[i]['y'][j]],
                                                  mode='lines',
                                                  # marker = dict(color='yellow'),
                                                  line=dict(width=5, color='green')
                                                  ))
    fig.layout.update(showlegend=False)
    fig.show()
    # Define plot space
    # fig, ax = plt.subplots(figsize=(10, 6))

    # Define x and y axes
    # ax.plot(testMinutePrices['t'],
    #       testMinutePrices['p'])

    # plt.show()
    return "DONE", bucketSize


def runBacktestCSV(data, noBuckets, zCalc, minutesInPrice, bucketTime):
    # first step calculate bucketSize, do this by sum(dailyVolume)/(1440/minutesInPrice)

    bucketSize = data['size'].sum() / (1440 / bucketTime)

    done = """testInputData, lastTime, length, data = readTestData(startTime, symbol, 7)
    a = 0.032
    testMinutePrices, testLeftOver = buildMinutePrices(testInputData, a, minsInPrice)
    buckets, leftOverBuckets = buildBuckets(testInputData, bucketSize)
    order = "Waiting"
    orderZ = 0
    direction = None
    orders = []
    orderPrice = 0
    orderSize = 1
    # read rest of the tick data and run algo
    while lastTime < length - 1:
        lastTime += 1
        # read through rest of the tick data and return each line separately, return each line
        _data = getLastTick(lastTime, data)
        testInputData = testInputData.append(_data, ignore_index=True)
        testInputData = testInputData.iloc[1:]
        testInputData, testMinutePrices, testLeftOver, updated = appMinutePrices(testInputData,
                                                                                 testMinutePrices,
                                                                                 testLeftOver, a, minsInPrice)
        buckets, leftOverBuckets = updateBuckets(testInputData, buckets,
                                                 leftOverBuckets, bucketSize)
        if updated:  # vpin
            if len(testMinutePrices) > 10:
                testMinutePrices = testMinutePrices.iloc[1:]
        if len(buckets) > 100:
            buckets = buckets.iloc[1:]
        z_score = calcVpin(leftOverBuckets, buckets, bucketSize, zCalc)
        trend = testMinutePrices['trend'].iloc[-1]
        if order == "Waiting":  # no current order
            #print(trend, z_score)
            if trend == 'uptrend' and z_score > 1.5:
                #print("Buy Signal")
                #print(trend, z_score)
                orderPrice = float(testInputData['p'].iloc[-1])
                orderSize = 1
                orderZ = z_score
                order = "Active"
                direction = "Buy"
                #orders = orders.append({'dir': direction, 'p': orderPrice, 's': orderSize, 'z': orderZ})
            elif trend == 'downtrend' and z_score < -1.5:
                #print("Sell Signal")
                #print(trend, z_score)
                orderPrice = float(testInputData['p'].iloc[-1])
                orderSize = 1
                orderZ = z_score
                order = "Active"
                direction = "Sell"
                #orders = orders.append({'dir': direction, 'p': orderPrice, 's': orderSize, 'z': orderZ})
        if order == "Active":  # current order placed
            #print(trend, z_score)
            if direction == "Buy":
                if trend == "downtrend" or z_score < 0:
                    #print("Close Buy Signal")
                    #print(trend, z_score)
                    closePrice = float(testInputData['p'].iloc[-1])
                    order = "Waiting"
                    orders.append({'dir': direction, 'pO': orderPrice, 'pC': closePrice, 's': orderSize, 'zO': orderZ, 'zC': z_score})
                    #print(orders)
                    direction = None
                    orderPrice = None
                    orderSize = None
                    orderZ = None
            if direction == "Sell":
                if trend == "uptrend" or z_score > 0:
                    #print("Close Sell Signal")
                    #print(trend, z_score)
                    closePrice = float(testInputData['p'].iloc[-1])
                    order = "Waiting"
                    orders.append({'dir': direction, 'pO': orderPrice, 'pC': closePrice, 's': orderSize, 'zO': orderZ, 'zC': z_score})
                    #print(orders)
                    direction = None
                    orderPrice = None
                    orderSize = None
                    orderZ = None

    orders = pd.DataFrame(orders)
    st = str(startTime)
    st = st[-6:]
    zCalc = str(zCalc)
    reg = re.compile(r'[\\/.:*?"<>|\r\n]+')
    valid_name = reg.findall(zCalc)
    if valid_name:
        for nv in valid_name:
            zCalc = zCalc.replace(nv, "_")
    name = 'backtestdata/' + str(symbol) + st + '.xlsx'
    name = str(name)
    sheetName = "b" + str(noBuckets) + "z" + str(zCalc) + "p" + str(minsInPrice) + "bT" + bucketTime
    writer = pd.ExcelWriter(name, engine='openpyxl', mode='a')
    orders.to_excel(writer, sheet_name=sheetName, index=False)
    writer.save()
    #print(testMinutePrices, testLeftOver, buckets, leftOverBuckets, bucketSize)
    bucketSize = bucketSize
    return "DONE", bucketSize




"""
    return


yes = """    writer = pd.ExcelWriter('demo.xlsx', engine='openpyxl')
    test ='''wb = pd.ExcelWriter('data' + symbol + '/' + str(serverTime) + '.xlsx', engine='xlsxwriter')
    wb.save()'''
    df.to_excel(writer, sheet_name='hello')
    writer.save()
    currentTime = int(df['T'].iloc[-1])
"""
# def get_buckets(df, bucketSize):
#   volumeBuckets = pd.DataFrame(columns=['Buy', 'Sell', 'Time'])
#  count = 0
# BV = 0
# SV = 0
# for index, row in df.iterrows():
#   newVolume = row['volume']
#  z = row['z']
# if bucketSize < count + newVolume:
#    BV = BV + (bucketSize - count) * z
#   SV = SV + (bucketSize - count) * (1 - z)
#  volumeBuckets = volumeBuckets.append({'Buy': BV, 'Sell': SV, 'Time': index}, ignore_index=True)
# count = newVolume - (bucketSize - count)
# if int(count / bucketSize) > 0:
#   for i in range(0, int(count / bucketSize)):
#      BV = (bucketSize) * z
#     SV = (bucketSize) * (1 - z)
#    volumeBuckets = volumeBuckets.append({'Buy': BV, 'Sell': SV, 'Time': index}, ignore_index=True)

# count = count % bucketSize
# BV = (count) * z
# SV = (count) * (1 - z)
# else:
#   BV = BV + (newVolume) * z
#  SV = SV + (newVolume) * (1 - z)
# count = count + newVolume
#
#   volumeBuckets = volumeBuckets.set_index('Time')
#  return volumeBuckets


# def calc_vpin(data, bucketSize, window):
#   volume = float(data['q'])
#  trades = float(data['p'])

# trades_1min = trades.diff(1).resample('1min', how='sum').dropna()
# volume_1min = volume.resample('1min', how='sum').dropna()
# sigma = trades_1min.std()
# z = trades_1min.apply(lambda x: norm.cdf(x / sigma))
# df = pd.DataFrame({'z': z, 'volume': volume_1min}).dropna()

# volumeBuckets = get_buckets(df, bucketSize)
# volumeBuckets['VPIN'] = abs(volumeBuckets['Buy'] - volumeBuckets['Sell']).rolling(window).mean() / bucketSize
# volumeBuckets['CDF'] = volumeBuckets['VPIN'].rank(pct=True)

# return volumeBuckets


# sym = ['C', 'BAC', 'USB', 'JPM', 'WFC']
# df = {}

# volume = {'ADAUSDT': 1000000}
# for s in sym:
#   print('Calculating VPIN')
#  df[s] = calc_vpin(sec_trades[s], volume[s], 50)
# print(s + ' ' + str(df[s].shape))

# avg = pd.DataFrame()
# print(avg.shape)
# metric = 'CDF'
# avg[metric] = np.nan
# for stock, frame in df.items():
#   frame = frame[[metric]].reset_index().drop_duplicates(subset='Time', keep='last').set_index('Time')
#  avg = avg.merge(frame[[metric]], left_index=True, right_index=True, how='outer', suffixes=('', stock))
# print(avg.shape)
# avg = avg.dropna(axis=0, how='all').fillna(method='ffill')

# avg.to_csv('CDF.csv')