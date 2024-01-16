from torch.utils.data import Dataset, DataLoader
import pandas as pd
from nstransformer.utils.timefeatures import time_features


class MyDataset(Dataset):
    def __init__(self, df_data, f_x, target, size=None, timeenc=0, freq='d', flag='1'):
        self.data = df_data
        self.timeenc = timeenc
        self.freq = freq
        self.flag = flag
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

    def __len__(self):
        return len(self.df_dates)

    def __getitem__(self, idx):
        if idx + self.seq_len + self.pred_len >= len(self.df_dates):
            idx = len(self.df_dates) - self.seq_len - self.pred_len - 1
        start_time = self.df_dates[idx]
        end_time = self.df_dates[idx + self.seq_len]
        r_begin = self.df_dates[idx + self.seq_len - self.label_len]
        r_end = self.df_dates[idx + self.seq_len + self.pred_len]


        features = self.data[self.data.date.apply(
            lambda x: (x >= start_time) & (x < end_time))].reset_index(drop=True)
        target = self.data[self.data.date.apply(
            lambda x: (x >= r_begin) & (x < end_time))].reset_index(drop=True)

        seq_x_mark = self.data_stamp[idx:idx + self.seq_len]
        seq_y_mark = self.data_stamp[idx + self.seq_len - self.label_len:idx + self.seq_len + self.pred_len]
        # for each date between start_time and end_time, we need to calculate the features
        seq_x_mark = time_features(pd.to_datetime(seq_x_mark['date'].values), freq=self.freq)
        seq_x_mark = seq_x_mark.transpose(1, 0)

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
    target = ['target', 'close30']
    data_loader = DataLoader(MyDataset(df, f_x, target), batch_size=1, shuffle=False)
    # print first 10 samples
    for i, data in enumerate(data_loader):
        print(i)
    return


if __name__ == '__main__':
    read_data()
