from tdx_func import *
df = pd.read_csv('data/daily_data_simple.csv')
#
# df['boduan'] = df.groupby('index_code').apply(boduan).reset_index(drop=True)
# print(df.head())
data = df[df.date == '2019-01-02'].reset_index(drop=True)
data.set_index('index_code', inplace=True)
flattened_data = data.T.stack().to_frame().T
print(flattened_data.head())
flattened_data.to_csv('data/flattened_data.csv', index=False)
# df = pd.read_csv('data/flattened_data.csv')
