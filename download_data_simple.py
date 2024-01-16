import os
import pandas as pd
import yfinance as yf
import datetime
import akshare as ak
import yfinance
from utils import *

pd.options.mode.copy_on_write = True

data_path = 'data'

# load 上证50, 沪深300, 中证500, 创业板指
index_codes = ['sh000016', 'sh000300', 'sh000905', 'sz399006']
today = datetime.date.today().strftime('%Y-%m-%d')
df = pd.DataFrame()
# get index daily info
if os.path.exists(os.path.join(data_path, 'index_daily_simple.csv')):
    df = pd.read_csv(os.path.join(data_path, 'index_daily_simple.csv'))
else:
    for index_code in index_codes:
        print('loading index code: ', index_code)
        df_tmp = ak.stock_zh_index_daily(symbol=index_code)
        df_tmp['index_code'] = index_code
        df = pd.concat([df, df_tmp], ignore_index=True)

    df_tmp = ak.index_us_stock_sina(symbol=".INX")
    df_tmp['index_code'] = 'sp500'
    df_tmp.drop(['amount'], axis=1, inplace=True)
    df = pd.concat([df, df_tmp], ignore_index=True)

    df_tmp = ak.index_us_stock_sina(symbol=".DJI")
    df_tmp['index_code'] = 'dowjones'
    df_tmp.drop(['amount'], axis=1, inplace=True)
    df = pd.concat([df, df_tmp], ignore_index=True)

    df_tmp = ak.index_us_stock_sina(symbol=".IXIC")
    df_tmp['index_code'] = 'nasdaq'
    df_tmp.drop(['amount'], axis=1, inplace=True)
    df = pd.concat([df, df_tmp], ignore_index=True)

    df_tmp = yfinance.Ticker("^GDAXI").history(start="2015-01-01", end=today, interval="1d")
    df_tmp = df_tmp.reset_index()
    df_tmp['index_code'] = 'dax'
    df_tmp.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
    df_tmp.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low',
                           'Close': 'close', 'Volume': 'volume'}, inplace=True)
    df_tmp['date'] = df_tmp['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    df = pd.concat([df, df_tmp], ignore_index=True)

    df_tmp = yfinance.Ticker("^FCHI").history(start="2015-01-01", end=today, interval="1d")
    df_tmp = df_tmp.reset_index()
    df_tmp['index_code'] = 'cac40'
    df_tmp.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
    df_tmp.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low',
                           'Close': 'close', 'Volume': 'volume'}, inplace=True)
    df_tmp['date'] = df_tmp['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    df = pd.concat([df, df_tmp], ignore_index=True)

    df_tmp = yfinance.Ticker("^N225").history(start="2015-01-01", end=today, interval="1d")
    df_tmp = df_tmp.reset_index()
    df_tmp['index_code'] = 'nikkei225'
    df_tmp.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
    df_tmp.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low',
                           'Close': 'close', 'Volume': 'volume'}, inplace=True)
    df_tmp['date'] = df_tmp['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    df = pd.concat([df, df_tmp], ignore_index=True)

    df_tmp = ak.spot_hist_sge(symbol='Au99.99')
    df_tmp['index_code'] = 'Au99'
    df_tmp['date'] = df_tmp['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    df = pd.concat([df, df_tmp], ignore_index=True)

    df.to_csv(os.path.join(data_path, 'index_daily_simple.csv'), index=False)

# give volume of au99 a value of 0
df.loc[df.index_code == 'Au99', 'volume'] = 0.01

# download cn and us bond data
if os.path.exists(os.path.join(data_path, 'bond_zh_us_rate.csv')):
    bond_zh_us_rate_df = pd.read_csv(os.path.join(data_path, 'bond_zh_us_rate.csv'))
else:
    bond_zh_us_rate_df = ak.bond_zh_us_rate(start_date="2015-01-01")
    bond_zh_us_rate_df.to_csv(os.path.join(data_path, 'bond_zh_us_rate.csv'), index=False)

# download cn rate_interbank
if os.path.exists(os.path.join(data_path, 'Shibor_rate_interbank.csv')):
    Shibor_rate_interbank_df = pd.read_csv(os.path.join(data_path, 'Shibor_rate_interbank.csv'))
    Chibor_rate_interbank_df = pd.read_csv(os.path.join(data_path, 'Chibor_rate_interbank.csv'))
else:
    Shibor_rate_interbank_df = ak.rate_interbank(
        market="上海银行同业拆借市场", symbol="Shibor人民币", indicator="1年")
    Shibor_rate_interbank_df.to_csv(os.path.join(data_path, 'Shibor_rate_interbank.csv'), index=False)
    Chibor_rate_interbank_df = ak.rate_interbank(
        market="中国银行同业拆借市场", symbol="Chibor人民币", indicator="1年")
    Chibor_rate_interbank_df.to_csv(os.path.join(data_path, 'Chibor_rate_interbank.csv'), index=False)

# download cpi data
if os.path.exists(os.path.join(data_path, 'cpi_china.csv')):
    macro_china_cpi_monthly_df = pd.read_csv(os.path.join(data_path, 'cpi_china.csv'))
    macro_usa_cpi_monthly_se = pd.read_csv(os.path.join(data_path, 'cpi_usa.csv'))
else:
    macro_china_cpi_monthly_df = ak.macro_china_cpi_monthly()
    macro_usa_cpi_monthly_se = ak.macro_usa_cpi_monthly()
    macro_usa_cpi_monthly_se = macro_usa_cpi_monthly_se.reset_index()
    macro_china_cpi_monthly_df = macro_china_cpi_monthly_df.reset_index()
    macro_usa_cpi_monthly_se.to_csv(os.path.join(data_path, 'cpi_usa.csv'), index=False)
    macro_china_cpi_monthly_df.to_csv(os.path.join(data_path, 'cpi_china.csv'), index=False)
# download ppi data
if os.path.exists(os.path.join(data_path, 'ppi.csv')):
    ppi_df = pd.read_csv(os.path.join(data_path, 'ppi.csv'))
else:
    ppi_df = ak.macro_china_ppi_yearly()
    ppi_df = ppi_df.reset_index()
    ppi_df.to_csv(os.path.join(data_path, 'ppi.csv'), index=False)

# download us overnight rate, 20+ year bond rate
if os.path.exists(os.path.join(data_path, 'overnight_rate.csv')):
    overnight_rate_df = pd.read_csv(os.path.join(data_path, 'overnight_rate.csv'))
    bond_20y_df = pd.read_csv(os.path.join(data_path, '20y_bond.csv'))
else:

    overnight_rate_df = yfinance.Ticker("XFFE.L").history(start="2015-01-01", end=today, interval="1d")
    overnight_rate_df = overnight_rate_df.reset_index()
    overnight_rate_df['date'] = overnight_rate_df['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    overnight_rate_df.drop(['Dividends', 'Stock Splits', 'Volume', 'Capital Gains', 'Date'], axis=1, inplace=True)
    overnight_rate_df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low',
                                        'Close': 'close'}, inplace=True)
    bond_20y_df = yfinance.Ticker("TLT").history(start="2015-01-01", end=today, interval="1d")
    bond_20y_df = bond_20y_df.reset_index()
    bond_20y_df['date'] = bond_20y_df['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    bond_20y_df.drop(['Dividends', 'Stock Splits', 'Volume', 'Capital Gains', 'Date'], axis=1, inplace=True)
    bond_20y_df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low',
                                    'Close': 'close'}, inplace=True)
    overnight_rate_df.to_csv(os.path.join(data_path, 'overnight_rate.csv'), index=False)
    bond_20y_df.to_csv(os.path.join(data_path, '20y_bond.csv'), index=False)
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
# find the code list which has all Nan values
list_2 = df_sum.T.sum()[df_sum.T.sum() > 0].index.to_list()
print('code list which has all Nan values:', list_2)
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

# TODO: add sma and ema

# calculate pct_change
df2['pre_close'] = df2.groupby('index_code')['close'].shift(1)
df2['pre_close'] = df2.groupby('index_code')['pre_close'].bfill()
df2['pct_change'] = (df2['close'] - df2['pre_close']) / df2['pre_close']
# calculate target
df2['close30'] = df2.groupby('index_code')['close'].shift(-30)
df2['close30'] = df2.groupby('index_code')['close30'].ffill()
df2['target'] = (df2['close30'] - df2['close']) / df2['close']
# df_nan = df2.isnull().any()
# print(df_nan)
# assert df_nan.sum() == 0

# drop index_code == cac40, dax, japan225
df2 = df2[df2.index_code != 'cac40'].reset_index(drop=True)
df2 = df2[df2.index_code != 'dax'].reset_index(drop=True)
df2 = df2[df2.index_code != 'nikkei225'].reset_index(drop=True)
df2 = df2[df2.index_code != 'Au99'].reset_index(drop=True)

# change date column from str to datetime type
df2['date'] = pd.to_datetime(df2['date'], format='%Y-%m-%d')
df2 = df2[df2.date >= pd.to_datetime('2017-01-01', format='%Y-%m-%d')].reset_index(drop=True)
df2 = df2[df2.date <= df2.groupby(by=['index_code']).date.max().min()].reset_index(drop=True)

# only keep those dates which all index codes have
date_list = df2.groupby(by=['date']).index_code.count()
date_list = date_list[date_list < date_list.max()].index.to_list()
print('percent of date need to be removed', len(date_list) / df2.shape[0])
df2 = df2[df2.date.apply(lambda x: x not in date_list)].reset_index(drop=True)

# check each index code has the same date
a = df2.groupby(by=['index_code']).date
# 对日期进行标记
df_dates = pd.DataFrame()
df_dates['trade_date'] = list(df2.date.sort_values().unique())
list_tmp = [0] * 29
list_tmp.append(1)
df_dates['target_date'] = list_tmp * (df_dates.shape[0] // 30) + [0] * (df_dates.shape[0] % 30)
# add bond data and rate_interbank data to df2
bond_zh_us_rate_df['日期'] = pd.to_datetime(bond_zh_us_rate_df['日期'], format='%Y-%m-%d')
bond_zh_us_rate_df.drop(['美国GDP年增率'], axis=1, inplace=True)
bond_zh_us_rate_df.drop(['中国GDP年增率'], axis=1, inplace=True)
df2 = pd.merge(df2, bond_zh_us_rate_df, how='left', left_on='date', right_on='日期')
df2.drop(['日期'], axis=1, inplace=True)
df2['中国国债收益率2年'] = df2['中国国债收益率2年'].ffill()
df2['中国国债收益率5年'] = df2['中国国债收益率5年'].ffill()
df2['中国国债收益率10年'] = df2['中国国债收益率10年'].ffill()
df2['中国国债收益率30年'] = df2['中国国债收益率30年'].ffill()
df2['美国国债收益率2年'] = df2['美国国债收益率2年'].ffill()
df2['美国国债收益率5年'] = df2['美国国债收益率5年'].ffill()
df2['美国国债收益率10年'] = df2['美国国债收益率10年'].ffill()
df2['美国国债收益率30年'] = df2['美国国债收益率30年'].ffill()
df2['中国国债收益率10年-2年'] = df2['中国国债收益率10年-2年'].ffill()
df2['美国国债收益率10年-2年'] = df2['美国国债收益率10年-2年'].ffill()

Chibor_rate_interbank_df['报告日'] = pd.to_datetime(Chibor_rate_interbank_df['报告日'], format='%Y-%m-%d')
Chibor_rate_interbank_df.rename(columns={'利率': '利率_Chibor', '涨跌': '涨跌_Chibor'}, inplace=True)
Shibor_rate_interbank_df['报告日'] = pd.to_datetime(Shibor_rate_interbank_df['报告日'], format='%Y-%m-%d')
Shibor_rate_interbank_df.rename(columns={'利率': '利率_Shibor', '涨跌': '涨跌_Shibor'}, inplace=True)
df2 = pd.merge(df2, Shibor_rate_interbank_df, how='left', left_on='date', right_on='报告日')
df2 = pd.merge(df2, Chibor_rate_interbank_df, how='left', left_on='date', right_on='报告日')
df2['利率_Shibor'] = df2['利率_Shibor'].ffill()
df2['涨跌_Shibor'] = df2['涨跌_Shibor'].ffill()
df2['利率_Chibor'] = df2['利率_Chibor'].ffill()
df2['涨跌_Chibor'] = df2['涨跌_Chibor'].ffill()
df2['利率_Shibor'] = df2['利率_Shibor'].bfill()
df2['涨跌_Shibor'] = df2['涨跌_Shibor'].bfill()
df2['利率_Chibor'] = df2['利率_Chibor'].bfill()
df2['涨跌_Chibor'] = df2['涨跌_Chibor'].bfill()

df2.drop(['报告日_x', '报告日_y'], axis=1, inplace=True)

# load Volatility data
if os.path.exists(os.path.join(data_path, 'Volatility.csv')):
    vix_df = pd.read_csv(os.path.join(data_path, 'Volatility.csv'))
    vix_df_50etf = pd.read_csv(os.path.join(data_path, 'Volatility_cn.csv'))
else:
    vix_df_50etf = ak.index_option_50etf_qvix()
    vix_df_50etf.to_csv(os.path.join(data_path, 'Volatility_cn.csv'), index=False)

    vix_df = yfinance.Ticker("^VIX").history(start="2005-01-01", end=today, interval="1d")
    vix_df = vix_df.reset_index()
    vix_df['Date'] = pd.to_datetime(vix_df['Date'], format='%Y-%m-%d').dt.tz_localize(None)
    vix_df.drop(['Dividends', 'Stock Splits', 'Volume'], axis=1, inplace=True)
    vix_df.to_csv(os.path.join(data_path, 'Volatility.csv'), index=False)

# merge Volatility data
vix_df['Date'] = pd.to_datetime(vix_df['Date'], format='%Y-%m-%d')
vix_df.rename(columns={'Open': 'open_vix_us',
                       'High': 'high_vix_us', 'Low': 'low_vix_us',
                       'Close': 'close_vix_us'}, inplace=True)
df2 = pd.merge(df2, vix_df, how='left', left_on='date', right_on='Date')
vix_df_50etf['date'] = pd.to_datetime(vix_df_50etf['date'], format='%Y-%m-%d')
vix_df_50etf.rename(columns={'open': 'open_vix_cn', 'high': 'high_vix_cn',
                             'low': 'low_vix_cn', 'close': 'close_vix_cn'}, inplace=True)
df2 = pd.merge(df2, vix_df_50etf, how='left', left_on='date', right_on='date')
df2.drop(['Date'], axis=1, inplace=True)
df2['open_vix_cn'] = df2['open_vix_cn'].ffill()
df2['high_vix_cn'] = df2['high_vix_cn'].ffill()
df2['low_vix_cn'] = df2['low_vix_cn'].ffill()
df2['close_vix_cn'] = df2['close_vix_cn'].ffill()

# merge cpi ppi data
macro_china_cpi_monthly_df['index'] = pd.to_datetime(macro_china_cpi_monthly_df['index'],
                                                     format='%Y-%m-%d')
macro_china_cpi_monthly_df.rename(columns={'index': 'date', 'cpi': 'cpi_cn'}, inplace=True)
df2 = pd.merge(df2, macro_china_cpi_monthly_df, how='left', left_on='date', right_on='date')
df2['cpi_cn'] = df2['cpi_cn'].ffill()
df2['cpi_cn'] = df2['cpi_cn'].bfill()
macro_usa_cpi_monthly_se['index'] = pd.to_datetime(macro_usa_cpi_monthly_se['index'],
                                                    format='%Y-%m-%d')
macro_usa_cpi_monthly_se.rename(columns={'index': 'date', 'cpi_monthly': 'cpi_us'}, inplace=True)
df2 = pd.merge(df2, macro_usa_cpi_monthly_se, how='left', left_on='date', right_on='date')
df2['cpi_us'] = df2['cpi_us'].ffill()
df2['cpi_us'] = df2['cpi_us'].bfill()

ppi_df['index'] = pd.to_datetime(ppi_df['index'], format='%Y-%m-%d')
ppi_df.rename(columns={'index': 'date', 'ppi': 'ppi_cn'}, inplace=True)
df2 = pd.merge(df2, ppi_df, how='left', left_on='date', right_on='date')
df2['ppi_cn'] = df2['ppi_cn'].ffill()
df2['ppi_cn'] = df2['ppi_cn'].bfill()

# merge overnight rate, 20+ year bond rate
overnight_rate_df['date'] = pd.to_datetime(overnight_rate_df['date'], format='%Y-%m-%d')
overnight_rate_df.rename(columns={'open': 'open_overnight_rate', 'high': 'high_overnight_rate',
                                    'low': 'low_overnight_rate', 'close': 'close_overnight_rate'}, inplace=True)
df2 = pd.merge(df2, overnight_rate_df, how='left', left_on='date', right_on='date')
df2['open_overnight_rate'] = df2['open_overnight_rate'].ffill()
df2['high_overnight_rate'] = df2['high_overnight_rate'].ffill()
df2['low_overnight_rate'] = df2['low_overnight_rate'].ffill()
df2['close_overnight_rate'] = df2['close_overnight_rate'].ffill()
bond_20y_df['date'] = pd.to_datetime(bond_20y_df['date'], format='%Y-%m-%d')
bond_20y_df.rename(columns={'open': 'open_20y_bond', 'high': 'high_20y_bond',
                                'low': 'low_20y_bond', 'close': 'close_20y_bond'}, inplace=True)
df2 = pd.merge(df2, bond_20y_df, how='left', left_on='date', right_on='date')
df2['open_20y_bond'] = df2['open_20y_bond'].ffill()
df2['high_20y_bond'] = df2['high_20y_bond'].ffill()
df2['low_20y_bond'] = df2['low_20y_bond'].ffill()
df2['close_20y_bond'] = df2['close_20y_bond'].ffill()

df_nan = df2.isnull().any()
assert df_nan.sum() == 0
# save data
df2.to_csv(os.path.join(data_path, 'daily_data_simple.csv'), index=False)
df_dates.to_csv(os.path.join(data_path, 'dates_simple.csv'), index=False)
