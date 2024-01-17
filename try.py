from tdx_func import *
df = pd.read_csv('data/daily_data_simple.csv')
#
# df['boduan'] = df.groupby('index_code').apply(boduan).reset_index(drop=True)
# print(df.head())
date = pd.read_csv('data/dates_simple.csv')
df_single = pd.DataFrame()
for date_single in date.trade_date:
    df_tmp = df[df.date == date_single].reset_index(drop=True)



data = df[df.date == '2019-01-02'].reset_index(drop=True)
data.set_index('index_code', inplace=True)
