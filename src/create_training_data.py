import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
from sklearn.preprocessing import MinMaxScaler

target_length=1
window_size_list = [5]
for window_size in window_size_list:

    for i, pair in enumerate(["CN50_USD", "EU50_EUR", "UK100_GBP", "US2000_USD","EUR_USD","USD_JPY",
                "JP225_USD","NAS100_USD","SPX500_USD","US30_USD"]):
        df = pd.read_csv('./stock_data/'+pair+"_M30.csv",header=0, index_col=0)
        if i == 0:
            all_df = df
        if i >= 1:
            all_df = pd.merge(all_df, df, on='Datetime', how='outer')
        all_df.sort_values(by="Datetime", ascending=True, inplace=True)
            
            
    all_df = all_df.query('USD_JPYOpen.notna() & EUR_USDOpen.notna()', engine='python')
    all_df = all_df.fillna(method='ffill')
    all_df = all_df.fillna(method='bfill')
    all_df.reset_index(drop=True, inplace=True)

    df = all_df
    CFD =["CN50_USD", "EU50_EUR", "UK100_GBP", "US2000_USD","EUR_USD","USD_JPY",
                "JP225_USD","NAS100_USD","SPX500_USD","US30_USD"]
    bases = ['Open', 'High', 'Low', 'Close']
    dfdate = df[['Datetime']]
    dfnotdate = df.drop(['Datetime'], axis=1).astype(float)
    df = pd.concat([dfdate, dfnotdate], axis=1)


    # 変化率を加える
    dfpct = pd.DataFrame()
    for cfd in CFD:
        dfpct[cfd+'Volumepct_change'] = (df[cfd+'Volume']+10).apply(np.log).pct_change()


    df_year_date_window_size = df.columns.get_loc('Datetime')
    df = pd.concat([df, dfpct], axis=1)
    df.dropna(how='any', inplace=True)
    loop_range = len(df)


    df_currency_window_size = []
    div_owari = []
    df_volume = []
    df_volume_change = []
    
    cols_to_find_cur = bases
    for c in CFD:
        df_currency_window_size.extend([df.columns.get_loc(c+col) for col in cols_to_find_cur])
        div_owari.append(df.columns.get_loc(c+'Close'))
        df_volume.append(df.columns.get_loc(c+'Volume'))
        df_volume_change.append(df.columns.get_loc(c+'Volumepct_change'))
    
    df_owari_currency = df.columns.get_loc('SPX500_USDClose')


    dfcolumn = []
    for i in range(window_size):
        dfcolumn.extend(['date'+str(i)])
        dfcolumn.extend(['open_base_SPX500'+str(i), 'high_base_SPX500'+str(i), 'low_base_SPX500'+str(i), 'close_base_SPX500'+str(i)])
        for cfd in CFD:
            dfcolumn.extend(['open'+cfd+str(i), 'high'+cfd+str(i), 'low'+cfd +str(i), 'close'+cfd+str(i)])
        for cfd in CFD:
            dfcolumn.extend(['vol'+cfd+str(i)])
        for cfd in CFD:
            dfcolumn.extend(['vol'+cfd+'_change'+str(i)])
        print(len(dfcolumn))
    dfcolumn.extend(['label'])
    print(len(dfcolumn))


    import csv
    num = df.values
    data = []
    data.append(dfcolumn)
    div=2
    df_stock_price=df.columns.get_loc('SPX500_USDClose')
    df_price_window_size=[]
    df_price_window_size.append(df.columns.get_loc('SPX500_USDOpen'))
    df_price_window_size.append(df.columns.get_loc('SPX500_USDHigh'))
    df_price_window_size.append(df.columns.get_loc('SPX500_USDLow'))
    df_price_window_size.append(df.columns.get_loc('SPX500_USDClose'))
    for i in range(loop_range):
        start = i
        end = i + window_size - 1
        if ((end+target_length) <= loop_range-1):
            num_window_size = num[start:end+1]
            num_window_size_final = np.empty((window_size, 0), float)
            num_window_size_final = np.append(num_window_size_final, num_window_size[:, df_year_date_window_size].reshape(window_size, -1), axis=1)
            num_window_size_final = np.append(num_window_size_final, num_window_size[:, df_price_window_size].reshape(window_size, -1), axis=1)
            for i in range(len(CFD)):
                div_currency_window_size = num_window_size[:, df_currency_window_size[i*4:(i+1)*4]]/num_window_size[window_size-div, div_owari[i]]-1
                num_window_size_final = np.append(num_window_size_final, div_currency_window_size, axis=1)

            for c in range(len(CFD)):
                num_window_size_final = np.append(num_window_size_final, num_window_size[:, df_volume[c]].reshape(window_size, -1), axis=1)
            for c in range(len(CFD)):
                num_window_size_final = np.append(num_window_size_final, num_window_size[:, df_volume_change[c]].reshape(window_size, -1), axis=1)
            numpy_window_size = num_window_size_final.ravel()

            label = num[end+target_length, df_owari_currency]/num[end, df_owari_currency]
            label = (label-1)*10000
            numpy_window_size = np.append(numpy_window_size, label)
            list_window_size = numpy_window_size.tolist()
            data.append(list_window_size)
            continue
        else:
            break

    with open('./training_data/training_data_'+str(window_size)+'.csv', 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(data)
