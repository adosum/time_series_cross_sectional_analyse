import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
from utils import *

# global parameters
len_train = 12

# load data
data_path = 'data'
df = pd.read_csv(os.path.join(data_path, 'daily_data.csv'))
df_dates = pd.read_csv(os.path.join(data_path, 'dates.csv'))
df = df.sort_values(by=['index_code', 'date']).reset_index(drop=True)

df_dates['trade_date'] = pd.to_datetime(df_dates['trade_date'], format='%Y-%m-%d')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

end_dates = df_dates[df_dates.target_date == 1].trade_date
end_dates = end_dates[pd.to_datetime('2013-01-01', format='%Y-%m-%d') < end_dates]
end_dates = end_dates[end_dates < pd.to_datetime('2023-01-01', format='%Y-%m-%d')]

# create model
model = linear_model.LinearRegression()
model_name = 'linear_regression'
f_x = ['close', 'open', 'high', 'low',
       'volume', 'pct_change',
       'macd', 'dea', 'dif', 'rsi', 'psy', 'bias']
dfa = pd.DataFrame()
for end_date in end_dates:
    end_index = df_dates[df_dates.trade_date == end_date].index.values[0]
    start_index = end_index - len_train * 20  # 20 days per month
    df_train_dates = df_dates[start_index:end_index + 1].reset_index(drop=True)
    start_time = df_train_dates[0:1].values[0][0]
    df_train = df[df.date.apply(lambda x: (x >= start_time) & (x < end_date))].reset_index(drop=True)
    df_test = df[df.date == end_date].reset_index(drop=True)
    print('预测时间：', end_date)
    assert df_train.shape[0] / df_test.shape[0] == len_train * 20
    assert df_train.target.isnull().sum() == 0
    mevtransformer = MedianExtremeValueTransformer()
    mevtransformer.fit(df_train[f_x])
    df_train[f_x] = mevtransformer.transform(df_train[f_x])
    df_test[f_x] = mevtransformer.transform(df_test[f_x])

    scaler = StandardScaler()
    scaler.fit(df_train[f_x])
    df_train[f_x] = scaler.transform(df_train[f_x])
    df_test[f_x] = scaler.transform(df_test[f_x])

    y_train = df_train['target'].copy()
    y_test = df_test['target'].copy()
    best_params = {}
    best_params[model_name] = {}
    axisx = np.arange(100, 2000, 100)

    model.fit(df_train[f_x], y_train)
    y_pred = model.predict(df_test[f_x]).flatten()
    # result
    df_result = df_test[['index_code', 'date']].copy()
    df_result['y'] = df_test['target']
    df_result['y_pred'] = y_pred
    df_result['y_pred_rank'] = df_result.y_pred.rank(ascending=False)
    df_result['class_label'] = df_result['y_pred_rank'].transform(
        lambda x: pd.qcut(x, 5, labels=range(1, 6), duplicates='drop'))

    df_result1 = df_result[df_result.class_label == 1]
    # calculate return
    dfa = pd.concat([dfa, df_result1.groupby(['date'])[['y']].mean()])
dfa = dfa.reset_index(drop=False)
# plot return
rates = [0, 2, 3, 5, 10]
for rate in rates:
    dfa['y{}'.format(rate)] = dfa.y + 1 - rate / 1000
    dfa['cum{}'.format(rate)] = dfa['y{}'.format(rate)].cumprod()
    x = dfa.date.values
    x = [str(x)[:10] for x in x]
    y1 = dfa.y.values - rate / 1000
    y2 = dfa['cum{}'.format(rate)].values
    label_x = 'date'
    # label_y1 = '周收益率'
    # label_y2 = '累计收益率'
    color_y1 = '#2A9CAD'
    color_y2 = "#FAB03D"
    title = 'montly return and accumulate trading fee ‰{}'.format(rate)

    fig, ax1 = plt.subplots(figsize=(10, 7), dpi=250)
    plt.xticks(rotation=60)
    ax2 = ax1.twinx()  # 做镜像处理

    lns1 = ax1.bar(x=x, height=y1, color=color_y1, alpha=0.7)
    lns2 = ax2.plot(x, y2, color=color_y2, ms=10)

    ax1.set_xlabel(label_x)  # 设置x轴标题

    ax1.grid(False)
    ax2.grid(False)

    # 设置横轴显示
    tick_spacing = 10  # 设置密度，比如横坐标9个，设置这个为3,到时候横坐标上就显示 9/3=3个横坐标，
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    # 添加标题
    plt.title(title)

    # 背景网格
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

dfa.to_csv(os.path.join(data_path, 'linear_regression_results.csv'), index=True)
