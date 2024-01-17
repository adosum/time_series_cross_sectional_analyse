from tdx_func import *
df = pd.read_csv('data/daily_data_simple.csv')
df['index_code'] = df['index_code'].map({0: 'sh000016', 1: 'sh000300', 2: 'sh000905',
                                            3: 'sz399006', 4: 'sp500', 5: 'dowjones',
                                            6: 'nasdaq', 7: 'dax', 8: 'cac40',
                                            9: 'nikkei225', 10: 'Au99'})
#
# df['boduan'] = df.groupby('index_code').apply(boduan).reset_index(drop=True)
# print(df.head())
date = pd.read_csv('data/dates_simple.csv')
df_single = pd.DataFrame()
for date_single in date.trade_date:
    flattened_data = df[df.date == date_single].reset_index(drop=True)
    flattened_data.set_index('index_code', inplace=True)
    flattened_data = flattened_data.T.stack().to_frame().T
    flattened_data.columns = flattened_data.columns.map('{0[0]}/{0[1]}'.format)
    df_single = pd.concat([df_single, flattened_data], axis=0)

columns = df_single.columns.tolist()

i = 0
while i < len(columns):
    for j in range(i+1, len(columns)):
        if df_single[columns[i]].equals(df_single[columns[j]]):
            df_single = df_single.drop(columns[j], axis=1)
            columns.pop(j)
            break
    else:
        i += 1

# move the target/sp500 to the last column
column_to_move = df_single.pop('target/sp500')
df_single.insert(len(df_single.columns), 'target/sp500', column_to_move)
df_single.to_csv('data/daily_data_simple_flattened.csv', index=False)

