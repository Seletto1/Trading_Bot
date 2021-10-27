import pandas as pd


def calcTrend(df, multiplier):
    """
    Function to calculate the price trend for each individual period
    :param df: DataFrame consisting of time(t), volume(v) and volume-weighted price(p) for each period
    :param multiplier: fixed value used in the calculation of the trend. Higher value leads to increased sensitivity to trend changes
    :return: DataFrame which is df with the trend direction for each row appended
    """
    minute_trends = {}
    df_copy = df.copy()
    for i in range(len(df)):  # loop through each of the rows in df
        difference = len(df) - 1 - i
        count = len(df) - 1 - difference
        if i % 10000 == 0:
            print(
                "count: {}, length: {}".format(i, len(minute_trends)))  # print out progress update every 10,000 periods
        if i < 2:
            minute_trends.update({df['t'][i]: {'it': 'N/A',
                                               'trend': 'notrend'}})  # for the first two values, no calculations are able to be completed
        elif i == 2:
            instantaneous = (multiplier - ((multiplier * multiplier) / 4.0)) * df_copy['p'].shift(count) + (
                    0.5 * multiplier * multiplier * df_copy['p'].shift(count - 1)) - (
                                    (multiplier - (0.75 * multiplier * multiplier)) * df_copy['p'].shift(
                                count - 2)) + (2 * (1 - multiplier) * ((df_copy['p'].shift(count) + 2 * df_copy[
                'p'].shift(count - 1) + df_copy['p'].shift(count - 2)) / 4.0)) - (
                                    (1 - multiplier) * (1 - multiplier) * ((df_copy['p'].shift(count) + 2 * df_copy[
                                'p'].shift(count - 1) + df_copy['p'].shift(count - 2)) / 4.0))
            minute_trends.update({df['t'][i]: {'it': float(instantaneous.iloc[count]), 'trend': 'notrend'}})
        elif i == 3:
            instantaneous = (multiplier - ((multiplier * multiplier) / 4.0)) * df_copy['p'].shift(count) + (
                    0.5 * multiplier * multiplier * df_copy['p'].shift(count - 1)) - (
                                    (multiplier - (0.75 * multiplier * multiplier)) * df_copy['p'].shift(
                                count - 2)) + (2 * (1 - multiplier) * minute_trends[df['t'][i - 1]]['it']) - (
                                    (1 - multiplier) * (1 - multiplier) * ((df_copy['p'].shift(count) + 2 * df_copy[
                                'p'].shift(count - 1) + df_copy['p'].shift(count - 2)) / 4.0))
            minute_trends.update({df['t'][i]: {'it': float(instantaneous.iloc[count]), 'trend': 'notrend'}})
        elif i > 3:
            instantaneous = (multiplier - ((multiplier * multiplier) / 4.0)) * df_copy['p'].iloc[i] + (
                    0.5 * multiplier * multiplier * df_copy['p'].iloc[i - 1]) - (
                                    (multiplier - (0.75 * multiplier * multiplier)) * df_copy['p'].iloc[i - 2]) + (
                                    2 * (1 - multiplier) * minute_trends[df_copy['t'].iloc[i - 1]]['it']) - (
                                    (1 - multiplier) * (1 - multiplier) * minute_trends[df_copy['t'].iloc[i - 2]]['it'])

            lag = instantaneous - float(minute_trends[df_copy['t'][i - 2]]['it'])
            # compare current instantaneous with the value 2 earlier to determine trend
            if lag > 0:
                minute_trends.update({df_copy['t'][i]: {'it': float(instantaneous), 'trend': 'uptrend'}})
            elif lag < 0:
                minute_trends.update({df_copy['t'][i]: {'it': float(instantaneous), 'trend': 'downtrend'}})
            else:
                minute_trends.update({df_copy['t'][i]: {'it': float(instantaneous), 'trend': 'notrend'}})

    minute_trends = pd.DataFrame(minute_trends)  # turn the dictionary of trends into pandas DataFrame
    minute_trends = minute_trends.transpose()
    df.set_index('t', inplace=True)
    minute_trends = df.join(minute_trends)  # join the trends to the initial DataFrame
    return minute_trends


def appendTrend(dictionary, multiplier, minute_trends, current_time):
    """
    When a specified period has passed in live data scraping, the trend for the most recent period needs to be calculated and returned
    :param dictionary: {time: {v:float, p:float}}
    :param multiplier: fixed value used in the calculation of the trend. Higher value leads to increased sensitivity to trend changes
    :param minute_trends: DataFrame with minute prices, volume and trend direction
    :param current_time: current time in milliseconds
    :return:
    """
    dfCopy = minute_trends.copy()
    volume = dictionary[current_time]['v']
    price = dictionary[current_time]['p']
    price2 = dfCopy['p'].iloc[-1]
    price3 = dfCopy['p'].iloc[-2]
    instantaneous = (multiplier - ((multiplier * multiplier) / 4.0)) * price + (
            0.5 * multiplier * multiplier * price2) - (
                            (multiplier - (0.75 * multiplier * multiplier)) * price3) + (
                            2 * (1 - multiplier) * dfCopy['it'].iloc[-1]) - (
                            (1 - multiplier) * (1 - multiplier) * dfCopy['it'].iloc[-2])
    lag = float(instantaneous) - float(dfCopy['it'].iloc[-2])
    if lag > 0:
        dictionary = {current_time: {'v': volume, 'p': price, 'it': float(instantaneous), 'trend': 'uptrend'}}
    elif lag < 0:
        dictionary = {current_time: {'v': volume, 'p': price, 'it': float(instantaneous), 'trend': 'downtrend'}}
    else:
        dictionary = {current_time: {'v': volume, 'p': price, 'it': float(instantaneous), 'trend': 'notrend'}}
    return dictionary
