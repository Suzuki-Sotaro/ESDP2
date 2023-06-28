import os
import numpy as np
import pandas as pd
import math
import datetime
import mplfinance as mpf
import shutil

def is_time_between(begin_time, end_time, check_time):
                check_time = check_time
                if begin_time < end_time:
                    return check_time >= begin_time and check_time <= end_time
                else:  # crosses midnight
                    return check_time >= begin_time or check_time <= end_time


def plot_result(X_TEST, Y_TEST, df_date_and_price, model, window_size, model_type):
    buylinelist = []
    selllinelist = []
    buysumpips = []
    sellsumpips = []
    buynum_alltrade = []
    sellnum_alltrade = []
    changetozero = 1000
    progfitminustrade = 0.1
    progfitminustrade2 = 0.1

    traiding_decision_list = [1]
    dir_name = './profit_result'+str(window_size)+'/'
    for traiding_decision in traiding_decision_list:
        predict = model.predict(X_TEST)
        df_plot = df_date_and_price[['date'+str(window_size-1), 'open_base_SPX500'+str(window_size-1), 'high_base_SPX500'+str(window_size-1), 'low_base_SPX500'+str(window_size-1), 'close_base_SPX500'+str(window_size-1)]].reset_index(drop=True)
        for order in ['BUY', 'SELL']:
            dfpredict = pd.DataFrame(data=predict, columns=['predict'], dtype='float')
            if order == 'BUY':
                line = 0.01*traiding_decision
                buylinelist.append(line)
            if order == 'SELL':
                line = -0.01*traiding_decision
                selllinelist.append(line)
            df_open_plot = df_plot['open_base_SPX500'+str(window_size-1)]
            # The buy/sell line is predict, so the opening price to buy/sell is shifted by one.
            df_open_plot1 = df_open_plot.shift(-1)      
            if order == 'BUY':  
                dfpredict.loc[dfpredict['predict'] > line, 'BUYPriceByPrediction'] = df_open_plot1
            elif order == 'SELL':
                dfpredict.loc[dfpredict['predict'] < line, 'SELLPriceByPrediction'] = df_open_plot1
            dfpredict[order+'PriceByPredictionAndClose'] = np.nan
            for index, row in dfpredict.iterrows():
                if index >= 1:
                    if (math.isnan(row[order+'PriceByPrediction'])) & ~(math.isnan(dfpredict.loc[index-1, order+'PriceByPrediction'])):
                        row[order+'PriceByPredictionAndClose'] = df_open_plot1[index]
            df_all_result = pd.concat([dfpredict['predict'], df_open_plot, df_open_plot1,
                                       dfpredict[order+'PriceByPrediction'], dfpredict[order+'PriceByPredictionAndClose'],
                                       df_plot['date'+str(window_size-1)]],  axis=1)

            
            for index, row in df_all_result.iterrows():
                if index >= 1:
                    date_str = row['date'+str(window_size-1)]
                    date_dt = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                    if date_dt.hour == 20 and date_dt.minute == 30:
                        df_all_result.loc[index, order+'PriceByPrediction'] = np.nan
                        if not math.isnan(df_all_result.loc[index-1, order+'PriceByPrediction']):
                            df_all_result.loc[index, order+'PriceByPredictionAndClose'] = df_open_plot1[index]

                    if is_time_between(datetime.time(20, 45), datetime.time(22, 30), date_dt.time()):
                        # print(row['date'+str(window_size-1)])
                        df_all_result.loc[index, order+'PriceByPrediction'] = np.nan
                        df_all_result.loc[index, order+'PriceByPredictionAndClose'] = np.nan

                    if date_dt.hour == 22 and date_dt.minute == 45:
                        df_all_result.loc[index, order+'PriceByPredictionAndClose'] = np.nan

                    
            df_all_result[order+'PriceByPredictionDiff'] = df_all_result[order+'PriceByPrediction'].fillna(0).diff()
            dfallpips = df_all_result[order+'PriceByPredictionDiff']+df_all_result[order+'PriceByPredictionAndClose'].fillna(0)
            dfallpips.rename('RESULT OF '+order, inplace=True)
            df_all_result = pd.concat([df_all_result, dfallpips], axis=1)
            df_all_result.loc[(df_all_result['RESULT OF '+order] > changetozero) | (df_all_result['RESULT OF '+order] < -changetozero), 'RESULT OF '+order] = 0
            df_all_result.set_index(pd.to_datetime(df_all_result['date'+str(window_size-1)]), inplace=True)
            df_all_result.drop(['date'+str(window_size-1)], axis=1, inplace=True)
            df_result_plot = df_all_result['RESULT OF '+order]  # resultをnp.nanで埋める前にaddplotの為に別で保存しておく
            df_result_plot_sum = df_all_result['RESULT OF '+order].rename('SUM')
            for i in range(len(df_result_plot_sum)):
                df_result_plot_sum[i] = df_result_plot[i]+df_result_plot[:i].sum()

            df_all_result.loc[df_all_result['RESULT OF '+order] == 0, 'RESULT OF '+order] = np.nan  # ここでnp.nanで埋める
            df_all_result = pd.concat([df_all_result, df_result_plot_sum], axis=1)
            df_all_result.to_csv(dir_name+str(model_type)+str(traiding_decision)+'で分けた時の'+order+'記録の全て.csv')
            sumpips = df_all_result['RESULT OF '+order].sum()
            if order == 'BUY':
                buysumpips.append(sumpips)
                buynum_alltrade.append(df_all_result['BUYPriceByPredictionAndClose'].count())
                dfbuyline = pd.concat([df_all_result['BUYPriceByPrediction'].reset_index(drop=True), df_plot['date'+str(window_size-1)]], axis=1)
                dfbuyline.set_index(pd.to_datetime(dfbuyline['date'+str(window_size-1)]), inplace=True)
                dfbuyline.drop(['date'+str(window_size-1)], axis=1, inplace=True)

                dfbuycloseline = pd.concat([df_all_result['BUYPriceByPredictionAndClose'].reset_index(drop=True), df_plot['date'+str(window_size-1)]], axis=1)
                dfbuycloseline.set_index(pd.to_datetime(dfbuycloseline['date'+str(window_size-1)]), inplace=True)
                dfbuycloseline.drop(['date'+str(window_size-1)], axis=1, inplace=True)

                df_buy_result_plot_sum = df_result_plot_sum

            if order == 'SELL':
                sellsumpips.append(sumpips)
                sellnum_alltrade.append(df_all_result['SELLPriceByPredictionAndClose'].count())

                dfsellline = pd.concat([df_all_result['SELLPriceByPrediction'].reset_index(drop=True), df_plot['date'+str(window_size-1)]], axis=1)
                dfsellline.set_index(pd.to_datetime(dfsellline['date'+str(window_size-1)]), inplace=True)
                dfsellline.drop(['date'+str(window_size-1)], axis=1, inplace=True)

                dfsellcloseline = pd.concat([df_all_result['SELLPriceByPredictionAndClose'].reset_index(drop=True), df_plot['date'+str(window_size-1)]], axis=1)
                dfsellcloseline.set_index(pd.to_datetime(dfsellcloseline['date'+str(window_size-1)]), inplace=True)
                dfsellcloseline.drop(['date'+str(window_size-1)], axis=1, inplace=True)

                df_sell_result_plot_sum = df_result_plot_sum
                
        df_plot.set_index(pd.to_datetime(df_plot['date'+str(window_size-1)]), inplace=True)
        df_plot.drop('date'+str(window_size-1), axis=1, inplace=True)
        df_plot = df_plot.astype(float)
        df_plot.columns = ['Open', 'High', 'Low', 'Close']
        cs = mpf.make_mpf_style(gridcolor="lightgray", rc={'font.size': 30})
        df_all_profit = df_buy_result_plot_sum - df_sell_result_plot_sum
        apd = [
            mpf.make_addplot(dfbuyline, scatter=True, markersize=10, marker='^', color="blue"),
            mpf.make_addplot(dfsellline, scatter=True, markersize=10, marker='v', color="red"),
            mpf.make_addplot(dfbuycloseline, scatter=True, markersize=10, marker='^', color="green"),
            mpf.make_addplot(dfsellcloseline, scatter=True, markersize=10, marker='v', color="green"),
            mpf.make_addplot(df_buy_result_plot_sum, color="blue"),
            mpf.make_addplot(df_sell_result_plot_sum, color="red"),
            mpf.make_addplot(df_all_profit, color="green"),
        ]
        # mpf.plot(df, addplot=apd, type='candle', volume=True, figratio=(10,5),
        filename = str(model_type)+str(traiding_decision)+'分割'+'.png'
        mpf.plot(df_plot, figratio=(24, 6), figscale=3, type='candle', show_nontrading=True, addplot=apd, style=cs,
                     savefig=dir_name+filename)

    df_buylinelist = pd.DataFrame(buylinelist, index=traiding_decision_list, columns=['BUY'])
    df_buysumpips = pd.DataFrame(buysumpips, index=traiding_decision_list, columns=['BUY_SUM_PIPS'])
    df_buynum_alltrade = pd.DataFrame(buynum_alltrade, index=traiding_decision_list, columns=['BUYnum_alltrade'])
    df_selllinelist = pd.DataFrame(selllinelist, index=traiding_decision_list, columns=['SELL'])
    df_sellsumpips = pd.DataFrame(sellsumpips, index=traiding_decision_list, columns=['SELL_SUM_PIPS'])
    df_sellnum_alltrade = pd.DataFrame(sellnum_alltrade, index=traiding_decision_list, columns=['SELLnum_alltrade'])

    df_lineslist = pd.concat([df_buylinelist, df_buysumpips, df_buynum_alltrade, df_selllinelist, df_sellsumpips, df_sellnum_alltrade], axis=1)
    df_lineslist['profit_pips'] = df_buysumpips['BUY_SUM_PIPS'] - df_sellsumpips['SELL_SUM_PIPS']
    df_lineslist['all_num_trade'] = df_buynum_alltrade['BUYnum_alltrade']+df_sellnum_alltrade['SELLnum_alltrade']
    df_lineslist['profit_pips_with_transcation_cost(%$0.5)'] = df_lineslist['profit_pips'] - df_lineslist['all_num_trade']*progfitminustrade
    df_lineslist['profit_pips_with_transcation_cost(%$0.1)'] = df_lineslist['profit_pips'] - df_lineslist['all_num_trade']*progfitminustrade2
    df_lineslist.to_csv(dir_name+str(model_type)+'Profit and Loss'+'.csv')
    
    return df_lineslist['profit_pips_with_transcation_cost(%$0.1)'].iloc[0]
