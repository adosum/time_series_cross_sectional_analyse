from torch.utils.data import Dataset, DataLoader
import pandas as pd
from nstransformer.utils.timefeatures import time_features


class MyDataset(Dataset):
    def __init__(self, df_data, f_x, target, size=None, timeenc=0, freq='d', flag='sp500'):
        self.data = df_data
        self.timeenc = timeenc
        self.freq = freq
        self.flag = flag
        self.flatten_data = self.flat_data(df_data)
        if size is not None:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        else:
            self.seq_len = 240
            self.label_len = 180
            self.pred_len = 10
        self.f_x = f_x
        self.target = target
        self.data_stamp = pd.DataFrame()
        self.data_stamp['date'] = df_data.date.sort_values().unique()
        self.data_stamp['date'] = pd.to_datetime(self.data_stamp.date)
        self.df_dates = list(df_data.date.sort_values().unique())
        if self.timeenc == 0:
            self.data_stamp['month'] = self.data_stamp.date.apply(lambda row: row.month, 1)
            self.data_stamp['day'] = self.data_stamp.date.apply(lambda row: row.day, 1)
            self.data_stamp['weekday'] = self.data_stamp.date.apply(lambda row: row.weekday(), 1)
            self.data_stamp['hour'] = self.data_stamp.date.apply(lambda row: row.hour, 1)
            self.data_stamp = self.data_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            self.data_stamp = time_features(pd.to_datetime(self.data_stamp['date'].values), freq=self.freq)
            self.data_stamp = self.data_stamp.transpose(1, 0)

    def flat_data(self, df):
        df['index_code'] = df['index_code'].map({0: 'sh000016', 1: 'sh000300', 2: 'sh000905',
                                                 3: 'sz399006', 4: 'sp500', 5: 'dowjones',
                                                 6: 'nasdaq', 7: 'dax', 8: 'cac40',
                                                 9: 'nikkei225', 10: 'Au99'})
        #
        # df['boduan'] = df.groupby('index_code').apply(boduan).reset_index(drop=True)
        # print(df.head())
        index_code = list(df.index_code.unique())
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
            for j in range(i + 1, len(columns)):
                if df_single[columns[i]].equals(df_single[columns[j]]):
                    df_single = df_single.drop(columns[j], axis=1)
                    columns.pop(j)
                    break
            else:
                i += 1

        target_name = 'target/' + self.flag
        column_to_move = df_single.pop(target_name)
        df_single.insert(len(df_single.columns), target_name, column_to_move)

        index_code.remove(self.flag)
        for name in index_code:
            target_name = 'target/' + name
            df_single.drop(target_name, axis=1, inplace=True)
        df_single.to_csv('data/daily_data_simple_flattened.csv', index=False)
        df_single = df_single.rename(columns={'date/sh000016': 'date'})
        return df_single

    def __len__(self):
        return len(self.df_dates)

    def __getitem__(self, idx):
        if idx + self.seq_len + self.pred_len >= len(self.df_dates):
            idx = len(self.df_dates) - self.seq_len - self.pred_len - 1
        start_time = self.df_dates[idx]
        end_time = self.df_dates[idx + self.seq_len]
        r_begin = self.df_dates[idx + self.seq_len - self.label_len]
        r_end = self.df_dates[idx + self.seq_len + self.pred_len]

        features = self.flatten_data[self.flatten_data.date.apply(
            lambda x: (x >= start_time) & (x < end_time))].reset_index(drop=True)
        target = self.flatten_data[self.flatten_data.date.apply(
            lambda x: (x >= r_begin) & (x < end_time))].reset_index(drop=True)
        seq_x_mark = self.data_stamp[idx:idx + self.seq_len]
        seq_y_mark = self.data_stamp[idx + self.seq_len:idx + self.seq_len + self.pred_len]

        return features[self.f_x].values, target[self.f_x].values, seq_x_mark, seq_y_mark


def read_data():
    df = pd.read_csv('data/daily_data_simple.csv')
    f_x = ['close', 'open', 'high', 'low',
           'volume', 'pct_change',
           'macd', 'dea', 'dif', 'rsi', 'psy', 'bias',
           '中国国债收益率2年', '中国国债收益率5年', '中国国债收益率10年', '中国国债收益率30年',
           '美国国债收益率2年', '美国国债收益率5年', '美国国债收益率10年', '美国国债收益率30年',
           '中国国债收益率10年-2年', '美国国债收益率10年-2年',
           '利率_Shibor', '涨跌_Shibor', '利率_Chibor',
           '涨跌_Chibor', 'open_vix_us', 'high_vix_us', 'low_vix_us', 'close_vix_us',
           'open_vix_cn', 'high_vix_cn', 'low_vix_cn', 'close_vix_cn', 'cpi_cn',
           'cpi_us', 'ppi_cn', 'open_overnight_rate', 'high_overnight_rate', 'low_overnight_rate',
           'close_overnight_rate',
           'open_20y_bond', 'high_20y_bond', 'low_20y_bond', 'close_20y_bond']
    target = ['target']
    data_loader = DataLoader(MyDataset(df, f_x, target), batch_size=1, shuffle=False)
    # print first 10 samples
    for i, data in enumerate(data_loader):
        print(i)
    return


if __name__ == '__main__':
    read_data()
