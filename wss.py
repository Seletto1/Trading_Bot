import time, sys, math
import numpy as np
import pandas as pd
import websocket
import threading
import json
from tradingpy import PRICEPRE, SIDE, Signal, QUANTPRE
from utilities import print_, orderstr, timestr, barstr
from binancepy import MarketData, Client


def wss_run(*args):
    #  threading functions
    def data_stream(*args):
        """
        First thread to send subscription to the exchange
        """
        params = [str.lower(ins) + str(s) for ins in insIds for s in stream]
        print_(params, fileout)
        ws.send(json.dumps({"method": "SUBSCRIBE", "params": params, "id": 1}))
        t1_idx = 0
        while len(endFlag) == 0:
            if len(SymKlns[insIds[0]]) % 5 == 0 and t1_idx < len(SymKlns[insIds[0]]) < models[insIds[0]].pdObserve:
                client.keepalive_stream()
                t1_idx = len(SymKlns[insIds[0]])

    def strategy(*args):
        """
        Second thread to generate signals upon the message from the exchange
        """
        t2_idx = {}
        for symbol in insIds:
            t2_idx[symbol] = 0
        while len(endFlag) == 0 and len(SymKlns[insIds[0]]) < models[insIds[0]].pdObserve:
            # try:
            for symbol in insIds:
                sym_ = SymKlns[symbol].copy()
                if len(sym_) > t2_idx[symbol]:
                    if models[symbol].modelType == 'bollinger':
                        data_ob = pd.DataFrame(sym_)
                        model_sig = models[symbol].get_last_signal(dataObserve=data_ob)
                    elif models[symbol].modelType == 'cradle':
                        data_ob = pd.DataFrame(sym_)
                        model_sig = models[symbol].get_last_signal(dataObserve=data_ob)
                    else:
                        model_sig = None
                    if model_sig is not None:
                        ready = True
                        if ready:
                            side, positionSide, startTime = model_sig['side'], model_sig['positionSide'], model_sig[
                                '_t'] + 60 * 1000
                            expTime, price = startTime + 10 * 60 * 1000, round(model_sig['_p'], PRICEPRE[symbol])  #
                            stopLoss = model_sig['sl']
                            takeProfit = model_sig['tp']
                            size = model_sig['size']
                            orderType, stopPrice = model_sig['type'], model_sig['stop']
                            stopPriceSL = model_sig['stopSL']
                            new_sig = Signal(symbol=symbol, side=side, size=size,
                                             orderType=orderType, price=price,
                                             startTime=startTime, expTime=expTime, stopLoss=stopLoss,
                                             takeProfit=takeProfit,
                                             timeLimit=models[symbol].pdEstimate * 60, timeInForce='GTC',
                                             stopPrice=stopPrice, stopPriceSL=stopPriceSL, positionSide=positionSide)
                            if in_position_(Signals[symbol], side='BOTH'):  # or position_count(insIds, Signals,
                                # side=side) >= \
                                # portfolio.equityDist[side]:
                                new_sig.set_expired()
                            else:
                                for sig in Signals[symbol]:
                                    if sig.is_waiting():
                                        sig.set_expired()
                                        print_('\n\tSet WAITING signal EXPIRED: \n\t' + str(sig), fileout)
                            Signals[symbol].append(new_sig)
                            print_('\n\tFOUND ' + str(new_sig), fileout)
                    t2_idx[symbol] = len(sym_)
        # except Exception:
        #   print_('\n\tClose on strategy()', fileout)
        #  ws.close()

    def book_manager(*args):
        """
        Third thread to excecute/cancel/track the signals generated in strategy()

        wb = Workbook()
        tim = str(int(time.time()))
        wb.save('report/' + tim + '.xlsx')
        wb.create_sheet('Sheet1', 0)
        ref = wb['Sheet']
        wb.remove(ref)
        active = wb.get_sheet_by_name('Sheet1')
        count = 1
        """
        while len(endFlag) == 0 and len(SymKlns[insIds[0]]) < models[insIds[0]].pdObserve:
            # try:
            # time.sleep(0.01)
            for symbol in insIds:
                in_position = False
                last_signal = None
                for sig in Signals[symbol]:
                    model = models[symbol]
                    sv_time = int(client.timestamp())
                    if sig.is_expired() or sig.is_closed():
                        Signals[symbol].remove(sig)
                    if sig.is_waiting():
                        # Check for EXPIRED order here
                        if sv_time > sig.expTime:
                            sig.set_expired()
                            print_('\n\tSet WAITING signal EXPIRED: \n\t' + str(sig), fileout)
                        else:
                            last_signal = sig
                    if sig.is_ordered():
                        order_update = client.query_order(symbol, sig.orderId)
                        slOrder = client.query_order(symbol, sig.orderSLId)
                        if order_update['status'] == 'FILLED':
                            order_update = client.query_order(symbol, sig.orderId)
                            in_position = True
                            if sig.side == 'BUY':
                                ####### quant = sig.get_quantity() / 2
                                quant = sig.get_quantity()
                                quant = round(quant, QUANTPRE[symbol])
                                takeProfit = sig.takeProfit + float(order_update['avgPrice'])
                                takeProfit = round(takeProfit, PRICEPRE[symbol])
                                # take profit order
                                if sig.positionSide == "BUY":
                                    posSide = "SHORT"
                                else:
                                    posSide = "LONG"
                                time.sleep(0.01)
                                orderTP = client.new_order(symbol=symbol, side="SELL",
                                                           orderType='TAKE_PROFIT_MARKET',
                                                           quantity=quant,
                                                           stopPrice=takeProfit, reduceOnly=True)
                                print(orderTP, "TP 143")
                                if len(orderTP) == 2:
                                    orderTP = client.new_order(symbol=symbol, side="SELL",
                                                               orderType='MARKET',
                                                               quantity=quant,
                                                               reduceOnly=True)
                                # priceSL = sig.stopPriceSl
                                # priceSL = round(priceSL, PRICEPRE[symbol])
                                # stopPriceSL = sig.stopLoss
                                # stopPriceSL = round(stopPriceSL, PRICEPRE[symbol])
                                # if sig.positionSide == "LONG":
                                #   posSide = "SHORT"
                                # else:
                                #    posSide = "LONG"
                                # orderSL = client.new_order(symbol=symbol, side="SELL", orderType='STOP',
                                # quantity=sig.get_quantity(),
                                # timeInForce='GTC', price=priceSL, stopPrice=stopPriceSL,
                                # reduceOnly=True)
                                # print(orderSL)
                                sig.set_active(excTime=order_update['updateTime'],
                                               excPrice=order_update['avgPrice'],
                                               excQty=order_update['executedQty'],
                                               orderIdTP=orderTP['orderId'],
                                               orderIdSL=sig.orderSLId)
                                sig.path_update(lastTime=sig.excTime, lastPrice=sig.excPrice)
                                print_('\n\tSet BOOKED order ACTIVE: \n\t' + str(sig) + '\n\t' + orderstr(
                                    order_update),
                                       fileout)
                            elif sig.side == 'SELL':
                                ###### quant = sig.get_quantity() / 2
                                quant = sig.get_quantity()
                                quant = round(quant, QUANTPRE[symbol])
                                takeProfit = float(order_update['avgPrice']) - sig.takeProfit
                                takeProfit = round(takeProfit, PRICEPRE[symbol])
                                # take profit order
                                if sig.positionSide == "BUY":
                                    posSide = "SHORT"
                                else:
                                    posSide = "LONG"
                                time.sleep(0.01)
                                orderTP = client.new_order(symbol=symbol, side="BUY",
                                                           orderType='TAKE_PROFIT_MARKET',
                                                           quantity=quant,
                                                           stopPrice=takeProfit, reduceOnly=True)
                                print(orderTP, "TP 181")
                                if len(orderTP) == 2:
                                    orderTP = client.new_order(symbol=symbol, side="BUY",
                                                               orderType='MARKET',
                                                               quantity=quant,
                                                               reduceOnly=True)
                                # priceSL = sig.stopPriceSl
                                # priceSL = round(priceSL, PRICEPRE[symbol])
                                # stopPriceSL = sig.stopLoss
                                # stopPriceSL = round(stopPriceSL, PRICEPRE[symbol])
                                # if sig.positionSide == "LONG":
                                #    posSide = "SHORT"
                                # else:
                                #    posSide = "LONG"
                                # orderSL = client.new_order(symbol=symbol, side="BUY", orderType='STOP',
                                #                           quantity=sig.get_quantity(),
                                #                           timeInForce='GTC', price=priceSL, stopPrice=stopPriceSL,
                                #                           reduceOnly=True)
                                # print(orderSL)
                                sig.set_active(excTime=order_update['updateTime'],
                                               excPrice=order_update['avgPrice'],
                                               excQty=order_update['executedQty'],
                                               orderIdTP=orderTP['orderId'],
                                               orderIdSL=sig.orderSLId)
                                sig.path_update(lastTime=sig.excTime, lastPrice=sig.excPrice)
                                print_('\n\tSet BOOKED order ACTIVE: \n\t' + str(sig) + '\n\t' + orderstr(
                                    order_update),
                                       fileout)
                        elif slOrder['status'] == "EXPIRED":
                            order_update = client.query_order(symbol, sig.orderId)
                            if order_update['status'] == "FILLED":
                                if sig.side == "BUY":
                                    side = "SELL"
                                else:
                                    side = "BUY"
                                order = client.new_order(symbol=symbol, side=side, orderType="MARKET",
                                                         quantity=sig.get_quantity(), reduceOnly=True)
                            elif order_update['status'] == 'PARTIALLY_FILLED':
                                order_update = client.query_order(symbol, sig.orderId)
                                client.cancel_order(symbol, sig.orderId)
                                if sig.side == 'BUY':
                                    quant = float(order_update['executedQty'])
                                    quant = round(quant, QUANTPRE[symbol])
                                    time.sleep(0.01)
                                    order = client.new_order(symbol=symbol, side="SELL",
                                                             orderType='MARKET',
                                                             quantity=quant,
                                                             reduceOnly=True)
                                elif sig.side == 'SELL':
                                    quant = float(order_update['executedQty'])
                                    quant = round(quant, QUANTPRE[symbol])
                                    time.sleep(0.01)
                                    orderTP = client.new_order(symbol=symbol, side="BUY",
                                                               orderType='MARKET',
                                                               quantity=quant, reduceOnly=True)
                            else:
                                client.cancel_order(symbol, sig.orderId)
                            sig.set_expired()
                            print_('\n\tSet order signal EXPIRED due to price movement: \n\t' + str(sig),
                                   fileout)
                        #  PROBLEM 3 Insert your code to handle EXPIRED and PARTIALLY_FILLED order here ###
                        elif sv_time > int(order_update['updateTime']) + 60 * 1000:
                            if order_update['status'] == 'PARTIALLY_FILLED':
                                in_position = True
                                order_update = client.query_order(symbol, sig.orderId)
                                client.cancel_order(symbol, sig.orderId)
                                if sig.side == 'BUY':
                                    ####quant = float(order_update['executedQty']) / 2
                                    quant = float(order_update['executedQty'])
                                    quant = round(quant, QUANTPRE[symbol])
                                    takeProfit = sig.takeProfit + float(order_update['avgPrice'])
                                    takeProfit = round(takeProfit, PRICEPRE[symbol])
                                    # take profit order
                                    if sig.positionSide == "BUY":
                                        posSide = "SHORT"
                                    else:
                                        posSide = "LONG"
                                    time.sleep(0.01)
                                    orderTP = client.new_order(symbol=symbol, side="SELL",
                                                               orderType='TAKE_PROFIT_MARKET',
                                                               quantity=quant,
                                                               stopPrice=takeProfit, reduceOnly=True)
                                    print(orderTP, "TP 257")
                                    if len(orderTP) == 2:
                                        orderTP = client.new_order(symbol=symbol, side="SELL",
                                                                   orderType='MARKET',
                                                                   quantity=quant,
                                                                   reduceOnly=True)
                                    sig.set_active(excTime=order_update['updateTime'],
                                                   excPrice=order_update['avgPrice'],
                                                   excQty=order_update['executedQty'],
                                                   orderIdTP=orderTP['orderId'])
                                    sig.path_update(lastTime=sig.excTime, lastPrice=sig.excPrice)
                                    print_('\n\tSet BOOKED order ACTIVE: \n\t' + str(sig) + '\n\t' + orderstr(
                                        order_update),
                                           fileout)
                                elif sig.side == 'SELL':
                                    #####quant = float(order_update['executedQty']) / 2
                                    quant = float(order_update['executedQty'])
                                    quant = round(quant, QUANTPRE[symbol])
                                    takeProfit = sig.takeProfit + float(order_update['avgPrice'])
                                    takeProfit = round(takeProfit, PRICEPRE[symbol])
                                    # take profit order
                                    if sig.positionSide == "BUY":
                                        posSide = "SHORT"
                                    else:
                                        posSide = "LONG"
                                    time.sleep(0.01)
                                    orderTP = client.new_order(symbol=symbol, side="BUY",
                                                               orderType='TAKE_PROFIT_MARKET',
                                                               quantity=quant,
                                                               stopPrice=takeProfit, reduceOnly=True)
                                    if len(orderTP) == 2:
                                        orderTP = client.new_order(symbol=symbol, side="BUY",
                                                                   orderType='MARKET',
                                                                   quantity=quant,
                                                                   reduceOnly=True)
                                    print(orderTP, "TP 281")
                                    sig.set_active(excTime=order_update['updateTime'],
                                                   excPrice=order_update['avgPrice'],
                                                   excQty=order_update['executedQty'],
                                                   orderIdTP=orderTP['orderId'])
                                    sig.path_update(lastTime=sig.excTime, lastPrice=sig.excPrice)
                                    print_('\n\tSet BOOKED order ACTIVE: \n\t' + str(sig) + '\n\t' + orderstr(
                                        order_update),
                                           fileout)
                            elif sv_time > order_update['updateTime'] + 10 * 60 * 1000:  # 10 mins after
                                client.cancel_order(symbol, sig.orderId)
                                sig.set_expired()
                                order_update = client.query_order(symbol, sig.orderId)
                                print_('\n\tSet BOOKED order EXPIRED: \n\t' + str(sig) + '\n\t' + orderstr(
                                    order_update), fileout)
                        # elif order_update['status'] != 'FILLED' or 'PARTIALLY_FILLED':
                        #   mkt = MarketData(testnet=testnet, symbol=symbol)
                        #  lastPriceDict = mkt.ticker_price_symbol(symbol=True)
                        # lastPrice = float(lastPriceDict['price'])
                        # if sig.side == "BUY" and lastPrice <= sig.stopLoss:
                        #   client.cancel_order(symbol, sig.orderId)
                        #  sig.set_expired()
                        # print_('\n\tSet order signal EXPIRED due to price movement: \n\t' + str(sig),
                        #       fileout)
                        # elif sig.side == "SELL" and lastPrice >= sig.stopLoss:
                        #   client.cancel_order(symbol, sig.orderId)
                        #  sig.set_expired()
                        # print_('\n\tSet order signal EXPIRED due to price movement: \n\t' + str(sig),
                        #       fileout)
                    if sig.is_active():
                        # Control ACTIVE position here
                        in_position = True
                        recent_trades = model.marketData.recent_trades(limit=2)
                        for trade in recent_trades:
                            if int(trade['time']) > sig.pricePath[-1]['timestamp']:
                                sig.path_update(lastTime=trade['time'], lastPrice=trade['price'])
                        exit_sign, pos = sig.exit_triggers()
                        order_update = client.query_order(symbol, sig.orderTPId)
                        print(order_update, "line319 tp ou")
                        order_updateTwo = client.query_order(symbol, sig.orderSLId)
                        print(order_updateTwo, "line321, sl ou")
                        if order_update['status'] == "FILLED":
                            client.cancel_order(symbol, sig.orderSLId)
                            print_('\n\tFound TAKE PROFIT', fileout)
                            trade = client.trade_list(symbol=symbol, limit=1)
                            print(trade, "trades")
                            quant = float(sig.get_quantity()) - float(order_update["executedQty"])
                            # quant = float(trade[-1]['qty'])
                            quant = round(quant, QUANTPRE[symbol])
                            rate = (sig.newStopLoss / sig.excPrice) * 100
                            rate = round(rate, 1)
                            if rate < 0.1:
                                rate = 0.1
                            elif rate > 5.0:
                                rate = 5
                            cnt_order = sig.counter_order()
                            if sig.positionSide == "BUY":
                                posSide = "SHORT"
                            else:
                                posSide = "LONG"
                            ######trailStop = client.new_order(symbol=symbol, side=cnt_order['side'],
                            #    orderType='TRAILING_STOP_MARKET', quantity=quant,
                            #   callbackRate=rate, reduceOnly=True)
                            ######print(trailStop, "ts")
                            #####sig.set_cnt_ordered(cntorderId=order_update['orderId'], cntType='MARKET',
                            #               cntTime=order_update['updateTime'],
                            #              trailorderId=trailStop['orderId'],
                            #             trailRate='priceRate', trigger="TP")
                            sig.set_closed(clsTime=order_update['updateTime'], clsPrice=float(order_update['price']))
                            print_('\tPlaced COUNTER order: \n\t' + str(sig) + '\n\t' + orderstr(order), fileout)
                        elif order_updateTwo['status'] == "FILLED":
                            client.cancel_order(symbol, sig.orderTPId)
                            print_('\n\tFound STOP LOSS', fileout)
                            pos = client.position_info()
                            for i in pos:
                                if i['symbol'] == str(symbol):
                                    amount = float(i['positionAmt'])
                                    if amount != 0:
                                        quant = -1 * amount
                                        quant = round(quant, QUANTPRE[symbol])
                                        if sig.side == "BUY":
                                            side = "SELL"
                                        else:
                                            side = "BUY"
                                        order = client.new_order(symbol=symbol, side=side,
                                                                 orderType='MARKET',
                                                                 quantity=quant,
                                                                 reduceOnly=True)
                                        print(order, "check if pos closed")
                                        sig.set_cnt_ordered(cntorderId=order['orderId'], cntType='MARKET',
                                                            cntTime=order['updateTime'],
                                                            trailorderId=None,
                                                            trailRate=None, trigger="SL")
                                    else:
                                        sig.set_cnt_ordered(cntorderId=sig.orderSLId, cntType='MARKET',
                                                            cntTime=order_updateTwo['updateTime'],
                                                            trailorderId=None,
                                                            trailRate=None, trigger="SL")
                            print_('\tPlaced COUNTER order: \n\t' + str(sig) + '\n\t' + orderstr(order), fileout)
                        elif exit_sign:
                            print_('\n\tFound ' + str(exit_sign), fileout)
                            cnt_order = sig.counter_order()
                            quant = cnt_order['amt']
                            quant = round(quant, QUANTPRE[symbol])
                            if sig.positionSide == "BUY":
                                posSide = "SHORT"
                            else:
                                posSide = "LONG"
                            order = client.new_order(symbol=symbol, side=cnt_order['side'], orderType='MARKET',
                                                     quantity=quant,
                                                     reduceOnly=True)
                            print(order, 'time')
                            sig.set_cnt_ordered(cntorderId=order['orderId'], cntType='MARKET',
                                                cntTime=order['updateTime'],
                                                trailorderId=None,
                                                trailRate=None, trigger="TIME")
                            print_('\tPlaced COUNTER order: \n\t' + str(sig) + '\n\t' + orderstr(order), fileout)
                        # else:
                        #   recent_trades = model.marketData.recent_trades(limit=1)
                        #  for trade in recent_trades:
                        #     print(trade)
                        #    print(recent_trades)
                        #   if sig.side == 'BUY':
                        #      if trade['price'] > order_update['price'] or trade['price'] < order_updateTwo['price']:
                        #         quant = sig.get_quantity()
                        #        closeorder = client.new_order(symbol=symbol, side="SELL", orderType='MARKET',
                        #                                     quantity=quant, reduceOnly=True)
                        #      client.cancel_order(symbol, sig.orderTPId)
                        #     client.cancel_order(symbol, sig.orderSLId)
                        #    sig.set_expired()
                        #   print("the error")
                        # elif sig.side == 'SELL':
                        #   if trade['price'] < order_update['price'] or trade['price'] > order_updateTwo['price']:
                        #      quant = sig.get_quantity()
                        #     closeorder = client.new_order(symbol=symbol, side="BUY", orderType='MARKET',
                        #                                  quantity=quant, reduceOnly=True)
                        #   client.cancel_order(symbol, sig.orderTPId)
                        #  client.cancel_order(symbol, sig.orderSLId)
                        # sig.set_expired()
                        # print("the error")

                    elif sig.is_cnt_ordered():
                        # Set CLOSED position here
                        in_position = True
                        if sig.trigger == "TP":  # take profit counter order, trail stop = placed
                            order_update = client.query_order(symbol, sig.orderTPId)
                            if order_update['status'] == 'FILLED':
                                order_updateTwo = client.query_order(symbol, sig.trailorderId)
                                order_updateThree = client.query_order(symbol, sig.orderSLId)
                                if order_updateTwo['status'] == 'FILLED':  # Trail filled, cancel stop loss
                                    client.cancel_order(symbol, sig.orderSLId)
                                    avePrice = (float(order_update['avgPrice']) + float(
                                        order_updateTwo['avgPrice'])) / 2
                                    sig.set_closed(clsTime=order_update['updateTime'], clsPrice=avePrice)
                                    print_('\n\tClosed order trail stop: \n\t' + str(sig) + '\n\t' + orderstr(
                                        order_update),
                                           fileout)
                                    orders = client.current_open_orders()
                                    for i in orders:
                                        if i['symbol'] == str(sig.symbol):
                                            client.cancel_order(symbol=sig.symbol, orderId=i['orderId'])
                                elif order_updateThree['status'] == 'FILLED':  # trail placed, stop triggered
                                    client.cancel_order(symbol, sig.trailorderId)  # cancel trail stop
                                    avePrice = ((float(order_update['executedQty']) * float(order_update['price'])) + (
                                            float(order_updateThree['executedQty']) * float(
                                        order_updateThree['price']))) / (float(
                                        order_update['executedQty']) + float(order_updateThree['executedQty']))
                                    sig.set_closed(clsTime=order_update['updateTime'], clsPrice=avePrice)
                                    print_('\n\tClosed order stop loss', fileout)
                                    orders = client.current_open_orders()
                                    for i in orders:
                                        if i['symbol'] == str(sig.symbol):
                                            client.cancel_order(symbol=sig.symbol, orderId=i['orderId'])
                        elif sig.trigger == "SL":  # stop loss, check position = 0
                            order_update = client.query_order(symbol, sig.orderSLId)
                            sig.set_closed(clsTime=order_update['updateTime'], clsPrice=float(order_update['price']))
                            print_('\n\tClosed order stop loss', fileout)
                            orders = client.current_open_orders()
                            for i in orders:
                                if i['symbol'] == str(sig.symbol):
                                    client.cancel_order(symbol=sig.symbol, orderId=i['orderId'])
                        elif sig.trigger == "TIME":  # close all open orders
                            client.cancel_order(symbol, sig.orderSLId)
                            client.cancel_order(symbol, sig.orderTPId)
                            order_update = client.query_order(symbol, sig.cntorderId)
                            if order_update['status'] == 'FILLED':
                                orders = client.current_open_orders()
                                for i in orders:
                                    if i['symbol'] == str(sig.symbol):
                                        client.cancel_order(symbol=sig.symbol, orderId=i['orderId'])
                            sig.set_closed(clsTime=order_update['updateTime'], clsPrice=float(order_update['price']))

                if (not in_position) and (last_signal is not None):
                    # Check for ENTRY and place NEW order here
                    sig = last_signal
                    if sig.orderType == 'MARKET':
                        order = client.new_order(symbol=symbol, side=sig.side, orderType=sig.orderType,
                                                 quantity=sig.get_quantity())
                        print(order)
                        sig.set_ordered(orderId=order['orderId'], orderTime=order['updateTime'], limitPrice=None)
                        print_('\n\tPlaced NEW order: \n\t' + str(sig) + '\n\t' + orderstr(order), fileout)
                    elif sig.orderType == 'LIMIT':
                        bids, asks, lim = get_possible_price(model.marketData, sig.side)
                        if lim is not None and (sig.price * 1.01 > lim > sig.price * 0.99):
                            order = client.new_order(symbol=symbol, side=sig.side, orderType=sig.orderType,
                                                     quantity=sig.get_quantity(),
                                                     timeInForce='GTC', price=lim)
                            print(order)
                            sig.set_ordered(orderId=order['orderId'], orderTime=order['updateTime'], limitPrice=lim)
                            print_('\n\tPlaced NEW order: \n\t' + str(sig) + '\n\t' + orderstr(order), fileout)
                    elif sig.orderType == "STOP":
                        lim = sig.stopPrice  # price = limit
                        lim = round(lim, PRICEPRE[symbol])
                        stopPrice = sig.price
                        stopPrice = round(stopPrice, PRICEPRE[symbol])
                        quantity = sig.get_quantity()
                        # main order
                        order = client.new_order(symbol=symbol, side=sig.side, orderType='STOP',
                                                 quantity=quantity,
                                                 timeInForce='GTC', price=lim, stopPrice=stopPrice)
                        time.sleep(0.01)
                        # stop loss order
                        if sig.side == "BUY":
                            side = "SELL"
                        elif sig.side == "SELL":
                            side = "BUY"
                        priceSL = sig.stopPriceSl
                        priceSL = round(priceSL, PRICEPRE[symbol])
                        stopPriceSL = sig.stopLoss
                        stopPriceSL = round(stopPriceSL, PRICEPRE[symbol])
                        if sig.positionSide == "LONG":
                            posSide = "SHORT"
                        else:
                            posSide = "LONG"
                        orderSL = client.new_order(symbol=symbol, side=side, orderType='STOP',
                                                   quantity=quantity,
                                                   timeInForce='GTC', price=priceSL, stopPrice=stopPriceSL,
                                                   reduceOnly=True)
                        if len(order) == 2 and len(orderSL) != 2:
                            client.cancel_order(symbol, orderSL['orderId'])
                            print("SET Expired due to order error")
                            sig.set_expired()
                        elif len(order) != 2 and len(orderSL) == 2:
                            client.cancel_order(symbol, order['orderId'])
                            print("SET Expired due to order error")
                            sig.set_expired()
                        elif len(order) == 2 and len(orderSL) == 2:
                            print("SET Expired due to order error")
                            sig.set_expired()
                        elif order["type"] != 'STOP':
                            client.cancel_order(symbol, orderSL['orderId'])
                            client.cancel_order(symbol, order['orderId'])
                            sig.set_expired()
                            print("SET Expired due to order error")
                        elif orderSL["type"] != 'STOP':
                            client.cancel_order(symbol, orderSL['orderId'])
                            client.cancel_order(symbol, order['orderId'])
                            sig.set_expired()
                            print("SET Expired due to order error")
                        else:
                            sig.set_ordered(orderId=order['orderId'], orderTime=order['updateTime'], limitPrice=lim,
                                            stopPrice=sig.stopPrice, orderSLId=orderSL['orderId'])
                            print_('\n\tPlaced NEW order: \n\t' + str(sig) + '\n\t' + orderstr(order), fileout)
        # except KeyError:
        # if order['msg'] == 'Order would immediately trigger.':  # if both initial orders fail
        # client.cancel_order(symbol, orderSL['orderId'])
        # print(order)
        # sig.set_expired()
        # print_('\n\tSet order signal EXPIRED due to Key Error: \n\t' + str(sig), fileout)
        #   if not sig.is_expired():  # if another key error
        #      print(KeyError)
        #     client.cancel_order(symbol, orderSL['orderId'])
        #    sig.set_expired()
        #   print_('\n\tSet order signal EXPIRED due to Key Error: \n\t' + str(sig), fileout)
        # except Exception:
        #   print_('\n\tClose on book_manager()', fileout)
        #  ws.close()
        ws.close()

    # websocket functions
    def on_message(ws, message):
        """
        Control the message received from
        """
        mess = json.loads(message)
        if mess['e'] == 'kline':
            kln = mess['k']
            if kln['x'] is True:
                symbol = kln['s'].upper()
                new_kln = {'_t': int(kln['t']), '_o': float(kln['o']), '_h': float(kln['h']), '_l': float(kln['l']),
                           '_c': float(kln['c']), '_v': float(kln['q'])}
                SymKlns[symbol].append(new_kln)
                print_('%d. %s\t' % (len(SymKlns[symbol]), symbol) + timestr(new_kln['_t']) + '\t' + \
                       ''.join(['{:>3}:{:<10}'.format(k, v) for k, v in iter(new_kln.items()) if not k == '_t']),
                       fileout)
        elif mess['e'] == 'aggTrade':
            symbol = mess['s']
            newTrade = {'a': int(mess['a']), 'p': float(mess['p']), 'q': float(mess['q']), 'f': int(mess['f']),
                        'l': int(mess['l']), 'T': int(mess['T']), 'm': bool(mess['m'])}
            SymKlns[symbol].append(newTrade)

            # print('Symbol: ' + str(symbol) + ', Time: '+str(new_trade['_t']) + ', ID: ' + str(new_trade['_id']) + ', Price: ' + \
            #    str(new_trade['_p']) + ', Quantity: ' + str(new_trade['_q']))

    def on_error(ws, error):
        """
        Do something when websocket has an error
        """
        print_(error, fileout)
        return

    def on_close(ws):
        """
        Do something when websocket closes
        """
        endFlag.append(1)
        for t in [t1, t2, t3]: t.join()
        return

    def on_open(ws, *args):
        """
        Start multi-threading functions
        """
        t1.start()
        t2.start()
        t3.start()
        return

    def position_count(insIds, signal_list, side='BOTH'):
        """
        Returns number of open positions
        """
        count = 0
        for s in insIds:
            for sig in signal_list[s]:
                if sig.side == side or side == 'BOTH':
                    if sig.is_ordered() or sig.is_active() or sig.is_cnt_ordered():
                        count += 1
        return count

    def in_position_(signal_list, side='BOTH'):
        """
        Check if there is any open positions
        """
        in_pos = False
        for sig in signal_list:
            if sig.side == side or side == 'BOTH':
                if sig.is_ordered() or sig.is_active() or sig.is_cnt_ordered():
                    in_pos = True
                    break
        return in_pos

    def get_possible_price(mk_data, side):
        """
        Return a safe limit price available on the market
        """
        mk_depth = mk_data.order_book(limit=5)
        bids = list(float(x[0]) for x in mk_depth['bids'])
        asks = list(float(x[0]) for x in mk_depth['asks'])
        try:
            lim = (side == 'BUY') * (bids[0] + bids[1]) / 2 + (side == 'SELL') * (asks[0] + asks[1]) / 2
            lim = round(lim, PRICEPRE[mk_data.symbol.upper()])
        except:
            lim = None
        return bids, asks, lim

    def sub_price_stream(ws, symbol):
        sym = str.lower(symbol)
        params = ['sym@miniTicker']
        print_(params, fileout)
        ws.send(json.dumps({"method": "SUBSCRIBE", "params": params, "id": 1}))
        return

    def unsub_price_stream(ws, symbol):
        sym = str.lower(symbol)
        params = ['sym@miniTicker']
        print_(params, fileout)
        ws.send(json.dumps({"method": "UNSUBSCRIBE", "params": params, "id": 312}))
        return

    start_time = time.time()
    portfolio, client, testnet, stream, models, fileout = args
    insIds = portfolio.tradeIns
    SymKlns = {}
    Signals = {}
    Order = {}
    expSigs = 0
    for symbol in insIds:
        SymKlns[symbol] = []
        Signals[symbol] = []

    endFlag = []
    t1 = threading.Thread(target=data_stream)
    t2 = threading.Thread(target=strategy)
    t3 = threading.Thread(target=book_manager)
    listen_key = client.get_listen_key()
    ws = websocket.WebSocketApp(f'{client.wss_way}{listen_key}',
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()
    client.close_stream()
    print_('\n' + barstr('Close Opening Positions', length=100, space_size=5) + '\n', fileout)
    # Close open positions
    in_position = False
    for symbol in insIds:
        if in_position_(Signals[symbol]):
            in_position = True
    while in_position:
        for symbol in insIds:
            model = models[symbol]
            for sig in Signals[symbol]:
                if sig.is_waiting():
                    sig.set_expired()
                    print_('\n\tSet WAITING signal EXPIRED: \n\t' + str(sig), fileout)
                elif sig.is_ordered():
                    client.cancel_order(symbol, sig.orderId)
                    client.cancel_order(symbol, sig.orderSLId)
                    sig.set_expired()
                    order_update = client.query_order(symbol, sig.orderId)
                    print_('\n\tSet BOOKED order EXPIRED: \n\t' + str(sig) + '\n\t' + orderstr(order_update), fileout)
                elif sig.is_active():
                    cnt_order = sig.counter_order()
                    order = client.new_order(symbol=symbol, side=cnt_order['side'], orderType='MARKET',
                                             quantity=cnt_order['amt'],
                                             reduceOnly=True)
                    sig.set_cnt_ordered(cntorderId=order['orderId'], cntType='MARKET',
                                        cntTime=order['updateTime'],
                                        trailorderId=None,
                                        trailRate=None)
                    print_('\tPlaced COUNTER order: \n\t' + str(sig) + '\n\t' + orderstr(order), fileout)
                elif sig.is_cnt_ordered():
                    order_update = client.query_order(symbol, sig.cntorderId)
                    if order_update['status'] == 'FILLED':
                        if sig.trailorderId is not None:
                            order_updateTwo = client.query_order(symbol, sig.trailorderId)
                            if order_updateTwo['status'] == 'FILLED':
                                avePrice = (float(order_update['avgPrice']) + float(order_updateTwo['avgPrice'])) / 2
                                sig.set_closed(clsTime=order_update['updateTime'], clsPrice=avePrice)
                                print_('\n\tClosed order trail stop: \n\t' + str(sig) + '\n\t' + orderstr(
                                    order_update),
                                       fileout)
                        elif sig.trailorderId is None:
                            sig.set_closed(clsTime=order_update['updateTime'],
                                           clsPrice=order_update['avgPrice'])
                            print_('\n\tClosed order: \n\t' + str(sig) + '\n\t' + orderstr(order_update),
                                   fileout)
        _position = False
        for symbol in insIds:
            if in_position_(Signals[symbol]):
                _position = True
                break
        in_position = _position

    orders = client.current_open_orders()
    for i in orders:
        client.cancel_order(i['symbol'], i['orderId'])
    print("cancelled all orders")
    return Signals, expSigs
