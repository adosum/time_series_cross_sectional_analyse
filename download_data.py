import os
import pandas as pd
import akshare as ak
from utils import *

pd.options.mode.copy_on_write = True

data_path = 'data'
# get index basic info
if os.path.exists(os.path.join(data_path, 'index_basic.csv')):
    df = pd.read_csv(os.path.join(data_path, 'index_basic.csv'))
else:
    df = ak.stock_zh_index_spot()
    df.to_csv(os.path.join(data_path, 'index_basic.csv'), index=False)

index_codes = df.代码.values
df = ak.stock_zh_index_daily(symbol=index_codes[0])
# add index code to df and remove first index code
df['index_code'] = index_codes[0]
index_codes = index_codes[1:]
# get index daily info
if os.path.exists(os.path.join(data_path, 'index_daily.csv')):
    df = pd.read_csv(os.path.join(data_path, 'index_daily.csv'))
else:
    for index_code in index_codes:
        print('loading index code: ', index_code)
        df_tmp = ak.stock_zh_index_daily(symbol=index_code)
        df_tmp['index_code'] = index_code
        df = pd.concat([df, df_tmp], ignore_index=True)
    df.to_csv(os.path.join(data_path, 'index_daily.csv'), index=False)

# calculate technical indicators
df['macd'] = (df.groupby(['index_code']).close.apply(lambda x: macd(x)).
              reset_index(drop=True))
df['dea'] = (df.groupby(['index_code']).close.apply(lambda x: dea(x))
             .reset_index(drop=True))
df['dif'] = (df.groupby(['index_code']).close.apply(lambda x: dif(x))
             .reset_index(drop=True))
df["rsi"] = (df.groupby(['index_code']).close.apply(lambda x: rsi(x))
             .reset_index(drop=True))
df["psy"] = (df.groupby(['index_code']).close.apply(lambda x: psy(x))
             .reset_index(drop=True))
df["bias"] = (df.groupby(['index_code']).close.apply(lambda x: bias(x))
              .reset_index(drop=True))
# remove duplicated rows
df = df[~df[['index_code', 'date']].duplicated()].reset_index(drop=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

df_sum = df.groupby(by=['index_code']).sum() == 0
# print('Nan value for each code:\n', df_sum.sum())
# find the code list which has all Nan values
list_2 = df_sum.T.sum()[df_sum.T.sum() > 0].index.to_list()
# remove the code list which has all Nan values
df2 = df[df.index_code.apply(lambda x: x not in list_2)]
print('percent of data removed:', (df.shape[0] - df2.shape[0]) / df.shape[0])
# fill Nan values with previous values for macd, dea, dif, rsi, psy, bias
df2['macd'] = df2.groupby(['index_code']).macd.bfill()
df2['dea'] = df2.groupby(['index_code']).dea.bfill()
df2['dif'] = df2.groupby(['index_code']).dif.bfill()
df2['rsi'] = df2.groupby(['index_code']).rsi.bfill()
df2['psy'] = df2.groupby(['index_code']).psy.bfill()
df2['bias'] = df2.groupby(['index_code']).bias.bfill()
df_nan = df2.isnull().any()
# calculate pct_change
df2['pre_close'] = df.groupby('index_code')['close'].shift(1)
df2['pct_change'] = (df2['close'] - df2['pre_close']) / df2['pre_close']
# calculate target
df2['close30'] = df.groupby('index_code')['close'].shift(-30)
df2['target'] = (df2['close30'] - df2['close']) / df2['close']
assert df_nan.sum() == 0

# change date column from str to datetime type
df2['date'] = pd.to_datetime(df2['date'], format='%Y-%m-%d')

df2 = df2[df2.date >= pd.to_datetime('2012-01-01', format='%Y-%m-%d')].reset_index(drop=True)
list1 = df2.groupby(by=['index_code']).date.count() != df2.groupby(by=['index_code']).date.count().describe().max()
# remove the index code which has false value
list1 = list1[list1 == True].index.to_list()
print('length of index code list which has not enough data:', len(list1))
df2 = df2[df2.index_code.apply(lambda x: x not in list1)]

# check each index code has the same date
# a = df2.groupby(by=['index_code']).date.describe()
# print(a)

# 对日期进行标记
df_dates = pd.DataFrame()
df_dates['trade_date'] = list(df2.date.sort_values().unique())
list_tmp = [0] * 29
list_tmp.append(1)
df_dates['target_date'] = list_tmp * (df_dates.shape[0] // 30) + [0] * (df_dates.shape[0] % 30)
# save data
df2.to_csv(os.path.join(data_path, 'daily_data.csv'), index=False)
df_dates.to_csv(os.path.join(data_path, 'dates.csv'), index=False)
