import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import datetime as dt
from datetime import timedelta
from datetime import date
from datetime import datetime as DT
from datetime import datetime
import argparse
import gzip
import glob
import os
import shutil
import time
import requests
from indicators import runBacktestCSV


def result(date_str):
    endpoint = 'https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/quote/{}.csv.gz'
    endpoint1 = 'https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/trade/{}.csv.gz'

    def req(x, y):
        r = requests.get(x.format(date_str))
        with open(date_str, 'wb') as fp:
            fp.write(r.content)

        with gzip.open(date_str, 'rb') as fp:
            data = fp.read()

        with open(date_str, 'wb') as fp:
            fp.write(data)
        try:
            df = pd.read_csv(format(date_str))
            if x == endpoint:
                print(endpoint)
                df.to_csv('q_' + format(date_str) + '.csv')
                os.remove(format(date_str))
            elif x == endpoint1:
                print(endpoint1)
                df.to_csv('t_' + format(date_str) + '.csv')
                os.remove(format(date_str))
        except:
            print("Something went wrong when writing to the file")

            # def rev():

    #    path = os.path.dirname(os.path.abspath(__file__))
    #    files = glob.glob(path+"/*.csv")
    #    os.remove(files[0])
    #    os.remove(files[2])

    #req(endpoint, date_str)
    #req(endpoint1, date_str)

    df1 = pd.read_csv('q_' + date_str + '.csv')
    df2 = pd.read_csv("t_" + date_str + ".csv")
    df2 = df2[df2['symbol'] == 'XBTUSD']
    df1 = df1[df1['symbol'] == 'XBTUSD']
    df2.reset_index(inplace=True)
    df1.reset_index(inplace=True)
    del df2['trdMatchID']
    del df2['homeNotional']
    del df2['foreignNotional']
    del df2['side']
    del df2['grossValue']

    df2['stock'] = df2['symbol'].loc[df2['symbol'] == 'XBTUSD'].fillna(method='ffill')

    df1.timestamp = df1.timestamp.map(lambda x: datetime.strptime(x, "%Y-%m-%dD%H:%M:%S.%f000"))
    data = """df1['date'] = (df1['timestamp'].str.split("D", expand=True)[0])
    df1['date'] = pd.to_datetime(df1['date'], infer_datetime_format=True)
    df1['time'] = (df1['timestamp'].str.split("D", expand=True)[1])
    df1['time'] = pd.to_timedelta(df1['time'])
    df1['DateTime'] = df1 ["date"] + df1["time"]
    """
    print(df1['timestamp'])

    stuff = """    df2['date'] = (df2['timestamp'].str.split("D", expand=True)[0])
    df2['date'] = pd.to_datetime(df2['date'], infer_datetime_format=True)
    df2.timestamp = df2.timestamp.map(lambda x: datetime.strptime(x, "%Y-%m-%dD%H:%M:%S.%f000"))

    df = df2[(df2['price'].notnull()) | (df1['bidPrice'].notnull())]
    df2['mtime'] = df2['timestamp'].apply(
        lambda x: x.hour * 3600 * 1000000 + x.minute * 60 * 1000000 + x.second * 1000000 + x.microsecond)
    df2['mtime_1m'] = df2['mtime'] + 10 * 1000000  # adding one minute in microseconds to mtime.
    df2['minute'] = df2['mtime'] // (60 * 1000000)  # adding one minute in microseconds to mtime.

    df = pd.merge_asof(left=df2, right=df1, on=['timestamp'], direction='nearest')
    print(df)

    df['Ask Size abs'] = df['askSize']
    df['Bid Size abs'] = df['bidSize']

    df['MidQuote'] = np.where((df['bidPrice'] != 0) & (df['askPrice'] != 0),
                              (df['bidPrice'] + df['askPrice']) / 2.0, np.nan)
    df2['MidQuote'] = df['MidQuote']
    df['prev_price'] = df.price.shift(+1)
    df['direction'] = np.where(df['price'] > df['MidQuote'], 'B', np.nan)
    df['direction'] = np.where(df['price'] < df['MidQuote'], 'S', df['direction'])
    df['direction'] = np.where(df['price'] == df['MidQuote'], 'C', df['direction'])
    dollar_value = (df['price'] * df['size'])
    df = df.assign(dollar_value=dollar_value)

    daily_value = df.groupby(['date', 'stock'])['dollar_value'].sum().reset_index()
    daily_value = daily_value.rename(columns={'value': 'daily_value'})
    df['quote_alive'] = df.groupby(['stock', 'date'])['mtime'].shift(-1) - df[
        'mtime']  # length of time between quotes

    df['quote_alive'] = df['quote_alive'].replace(0, np.nan)
    df['Quoted Spread'] = np.where(
        (df['askPrice'] != 0) & (df['bidPrice'] != 0) & (df['askPrice'] > df['bidPrice']),
        df['askPrice'] - df['bidPrice'], np.nan)

    df['Quoted Spread bps'] = np.where(
        (df['askPrice'] != 0) & (df['bidPrice'] != 0) & (df['askPrice'] > df['bidPrice']),
        (df['askPrice'] - df['bidPrice']) / (df['MidQuote']), np.nan)

    df['Quoted Spread_TW'] = df['Quoted Spread'] * df['quote_alive']
    df['Quoted Spread bps_TW'] = df['Quoted Spread bps'] * df['quote_alive']
    b_sel = df.direction == 'B'
    df.loc[b_sel, f'Effective Spread'] = 2 * (df.loc[b_sel, f'price'] - df.loc[b_sel, 'MidQuote'])  ####

    s_sel = df.direction == 'S'
    df.loc[s_sel, f'Effective Spread'] = 2 * (df.loc[s_sel, 'MidQuote'] - df.loc[
        s_sel, f'price'])  ##### why is it f'price' here but just 'price' for 'B'??
    df[f'Effective Spread_VW'] = df[f'Effective Spread'] * df[f'dollar_value']

    df.loc[b_sel, f'Effective Spread bps'] = 2 * (df.loc[b_sel, f'price'] - df.loc[b_sel, 'MidQuote']) / df.loc[
        b_sel, 'MidQuote']
    df.loc[s_sel, f'Effective Spread bps'] = 2 * (df.loc[s_sel, 'MidQuote'] - df.loc[s_sel, f'price']) / df.loc[
        s_sel, 'MidQuote']

    df[f'Effective Spread bps_VW'] = df[f'Effective Spread bps'] * df[f'dollar_value']
    trade_size = df.groupby(['date', 'stock'])['dollar_value'].mean().reset_index()
    trade_size = trade_size.rename(columns={'value': 'trade size'})
    Quoted_spread_TW = (df.groupby(['stock', 'minute'])['Quoted Spread_TW'].sum() / \
                        df.groupby(['stock', 'minute'])['quote_alive'].sum()).reset_index()
    Quoted_spread_TW = Quoted_spread_TW.rename(columns={0: 'quoted spread'})
    df['MidQuote_TW'] = df['MidQuote'] * df['quote_alive']
    midpoint_TW = (df.groupby(['stock', 'minute'])['MidQuote_TW'].sum() / \
                   df.groupby(['stock', 'minute'])['quote_alive'].sum()).reset_index()
    midpoint_TW = midpoint_TW.rename(columns={0: 'midpoint'})
    Quoted_spread_bps_TW = (df.groupby(['stock', 'minute'])['Quoted Spread bps_TW'].sum() / \
                            df.groupby(['stock', 'minute'])['quote_alive'].sum()).reset_index()
    Quoted_spread_bps_TW = Quoted_spread_bps_TW.rename(columns={0: 'quoted spread bps'})
    Value_weighted_Effective_Spread = (df.groupby(['stock', 'minute'])['Effective Spread_VW'].sum() / \
                                       df[df['Effective Spread_VW'].notnull()].groupby(['stock', 'minute'])[
                                           'dollar_value'].sum()).reset_index()
    Value_weighted_Effective_Spread = Value_weighted_Effective_Spread.rename(
        columns={0: 'value weighted effective spread'})
    Value_weighted_Effective_Spread_bps = ((df.groupby(['stock', 'minute'])['Effective Spread bps_VW'].sum() / \
                                            df[df['Effective Spread bps_VW'].notnull()].groupby(['stock', 'minute'])[
                                                'dollar_value'].sum()) * 10000).reset_index()
    Value_weighted_Effective_Spread_bps = Value_weighted_Effective_Spread_bps.rename(
        columns={0: 'value weighted effective spread bps'})
    df = df.sort_values('mtime', ascending=True)
    df_realized = pd.merge_asof(df, df2[['stock', 'date', 'mtime', 'MidQuote']].sort_values('mtime'),
                                left_on=['mtime_1m'], right_on=['mtime'], by=['stock', 'date'],
                                suffixes=('', '_1m_matched'), allow_exact_matches=False)
    print(df_realized)

    df_realized.plot(y=['MidQuote_1m_matched', 'MidQuote'], figsize=(20, 12))

    df_realized['priceImpact'] = 2 * (df_realized['MidQuote_1m_matched'] - df_realized['MidQuote'])
    b_sel = df_realized.direction == 'B'
    df_realized.loc[b_sel, f'Realized Spread'] = df.loc[b_sel, f'Effective Spread'] - df_realized.loc[
        b_sel, 'priceImpact']
    s_sel = df_realized.direction == 'S'
    df_realized.loc[s_sel, f'Realized Spread'] = df.loc[s_sel, f'Effective Spread'] - df_realized.loc[
        s_sel, 'priceImpact']
    df_realized[f'Realized Spread_VW'] = df_realized[f'Realized Spread'] * df_realized[f'dollar_value']
    df_realized.loc[b_sel, f'Realized Spread bps'] = 2 * (
            df_realized.loc[b_sel, f'price'] - df_realized.loc[b_sel, 'MidQuote_1m_matched']) / df_realized.loc[
                                                         b_sel, 'MidQuote_1m_matched']
    df_realized.loc[s_sel, f'Realized Spread bps'] = 2 * (
            df_realized.loc[s_sel, 'MidQuote_1m_matched'] - df_realized.loc[s_sel, f'price']) / df_realized.loc[
                                                         s_sel, 'MidQuote_1m_matched']
    df_realized[f'Realized Spread bps_VW'] = df_realized[f'Realized Spread bps'] * df_realized[f'dollar_value']
    Value_weighted_Realized_Spread = (df_realized.groupby(['stock', 'minute'])['Realized Spread_VW'].sum() / \
                                      df_realized[df_realized['Realized Spread_VW'].notnull()].groupby(
                                          ['stock', 'minute'])['dollar_value'].sum()).reset_index()
    Value_weighted_Realized_Spread = Value_weighted_Realized_Spread.rename(
        columns={0: 'value weighted realized spread'})
    Value_weighted_Realized_Spread_bps = ((df_realized.groupby(['stock', 'minute'])['Realized Spread bps_VW'].sum() / \
                                           df_realized[df_realized['Realized Spread bps_VW'].notnull()].groupby(
                                               ['stock', 'minute'])['dollar_value'].sum()) * 10000).reset_index()
    Value_weighted_Realized_Spread_bps = Value_weighted_Realized_Spread_bps.rename(
        columns={0: 'value weighted realized spread bps'})
    realised_spread_xbt = df_realized.loc[df_realized['stock'] == 'XBTUSD']
    realised_spread_xbt = realised_spread_xbt[pd.notnull(realised_spread_xbt['Effective Spread'])]
    Realised_spread_table = Value_weighted_Realized_Spread.assign(
        Value_weighted_Realized_Spread_bps=Value_weighted_Realized_Spread_bps['value weighted realized spread bps'],
        Quoted_spread_TW=Quoted_spread_TW['quoted spread'],
        Quoted_spread_bps_TW=Quoted_spread_bps_TW['quoted spread bps'],
        Value_weighted_Effective_Spread=Value_weighted_Effective_Spread['value weighted effective spread'],
        Value_weighted_Effective_Spread_bps=Value_weighted_Effective_Spread_bps['value weighted effective spread bps'])
    # Realised_spread_table.to_csv("realizespreadxbt.csv", mode = 'a')
    if not os.path.isfile('realizespreadxbt.csv'):
        Realised_spread_table.to_csv('realizespreadxbt.csv', header='column_names')
    else:  # else it exists so append without writing the header
        Realised_spread_table.to_csv('realizespreadxbt.csv', mode='a', header=False)
"""
    ### calc vpin for each minute
    #calc bucket size

    runBacktestCSV(data=df2, noBuckets=250, zCalc=0.5, minutesInPrice=1, bucketTime=5)

    # rev()


def scrape(year, date, end):
    end_date = min(DT(year, 12, 31), DT.today() - timedelta(days=1))

    while date <= end_date and date <= end:
        date_str = date.strftime('%Y%m%d')
        print("Processing {}...".format(date))
        print(format(date_str))
        result(date_str)

        date += timedelta(days=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BitMex historical data scraper. Scrapes files into single year CSVs')
    parser.add_argument('--start', default="20190901",
                        help='start date, in YYYYMMDD format. Default is 2014-11-22, the earliest data date for BitMex')
    parser.add_argument('--end', default="20190901", help='end date, in YYYYMMDD format. Default is yesterday')
    args = parser.parse_args()

    start = DT.strptime(args.start, '%Y%m%d')
    end = DT.strptime(args.end, '%Y%m%d') if args.end else dt.utcnow()

    years = list(range(start.year, end.year + 1))

    starts = [DT(year, 1, 1) for year in years]
    starts[0] = start

    for year, start in zip(years, starts):
        scrape(year, start, end)

    print("Done")
