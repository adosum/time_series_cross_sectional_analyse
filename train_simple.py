import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from utils import *

# remove files in results folder
for file in os.listdir('results'):
    file_path = os.path.join('results', file)
    if os.path.isfile(file_path):
        os.remove(file_path)

# remove files in tmp results folder
for file in os.listdir('tmp_results'):
    file_path = os.path.join('tmp_results', file)
    if os.path.isfile(file_path):
        os.remove(file_path)

# global parameters
len_train = 12
today = datetime.date.today().strftime('%Y-%m-%d')
# load data
data_path = 'data'
df = pd.read_csv(os.path.join(data_path, 'daily_data_simple.csv'))
df_dates = pd.read_csv(os.path.join(data_path, 'dates_simple.csv'))
df = df.sort_values(by=['index_code', 'date']).reset_index(drop=True)

df_dates['trade_date'] = pd.to_datetime(df_dates['trade_date'], format='%Y-%m-%d')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

end_dates = df_dates[df_dates.target_date == 1].trade_date
end_dates = end_dates[pd.to_datetime('2019-01-01', format='%Y-%m-%d') < end_dates]
# end_dates = end_dates[end_dates < pd.to_datetime('2023-01-01', format='%Y-%m-%d')]
end_dates = end_dates[end_dates < today]

# create model
model = XGBRegressor()
classification_model = XGBClassifier()
model_name = 'linear_regression'
f_x = ['close', 'open', 'high', 'low',
       'volume', 'pct_change',
       'macd', 'dea', 'dif', 'rsi', 'psy', 'bias',
       '中国国债收益率2年', '中国国债收益率5年', '中国国债收益率10年', '中国国债收益率30年',
       '美国国债收益率2年', '美国国债收益率5年', '美国国债收益率10年', '美国国债收益率30年',
       '中国国债收益率10年-2年', '美国国债收益率10年-2年',
       '利率_Shibor', '涨跌_Shibor', '利率_Chibor',
       '涨跌_Chibor', 'open_vix_us', 'high_vix_us', 'low_vix_us', 'close_vix_us',
       'open_vix_cn', 'high_vix_cn', 'low_vix_cn', 'close_vix_cn', 'cpi_cn',
       'cpi_us', 'ppi_cn', 'open_overnight_rate', 'high_overnight_rate', 'low_overnight_rate', 'close_overnight_rate',
       'open_20y_bond', 'high_20y_bond', 'low_20y_bond', 'close_20y_bond', 'index_code']
df['volume'] = df['volume'].apply(lambda x: np.log(x))

dfa = pd.DataFrame()
dfa_w = pd.DataFrame()
result_all = pd.DataFrame()
for end_date in end_dates:
    end_index = df_dates[df_dates.trade_date == end_date].index.values[0]
    start_index = end_index - len_train * 20  # 20 days per month
    df_train_dates = df_dates[start_index:end_index + 1].reset_index(drop=True)
    start_time = df_train_dates[0:1].values[0][0]
    df_train = df[df.date.apply(lambda x: (x >= start_time) & (x < end_date))].reset_index(drop=True)
    df_test = df[df.date == end_date].reset_index(drop=True)
    print('预测时间：', end_date)
    print(df_train.shape[0] / df_test.shape[0])
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
    # regression
    y_train = df_train['target'].copy()
    y_test = df_test['target'].copy()
    model.fit(df_train[f_x], y_train)
    y_pred = model.predict(df_test[f_x]).flatten()
    # classification
    y_train_class = y_train.apply(lambda x: 1 if x > 0 else 0)
    y_test_class = y_test.apply(lambda x: 1 if x > 0 else 0)
    classification_model.fit(df_train[f_x], y_train_class)
    y_pred_class = classification_model.predict(df_test[f_x]).flatten()

    # result
    df_result = df_test[['index_code', 'date']].copy()
    df_result['y'] = df_test['target']
    df_result['y_pred'] = y_pred
    df_result['y_pred_rank'] = df_result.y_pred.rank(ascending=False)
    df_result['class_label'] = df_result['y_pred_rank'].transform(
        lambda x: pd.qcut(x, 3, labels=range(1, 4), duplicates='drop'))
    df_result['y_pred_class'] = y_pred_class

    df_result1 = df_result[df_result.y_pred_class == 1]
    df_result1 = df_result1[df_result1.class_label == 1]
    df_result1.to_csv(os.path.join('results', str(end_date) + '.csv'), index=False)
    result_all = pd.concat([result_all, df_result1])
    # calculate weighted return
    df_result1['weight'] = 1 / df_result1['y_pred_rank']
    weight_sum = df_result1['weight'].sum()
    df_result1['weight'] = df_result1['weight'] / weight_sum
    df_result1['weighted_y'] = df_result1['y'] * df_result1['weight']

    # calculate return
    dfa = pd.concat([dfa, df_result1.groupby(['date'])[['y']].mean()])
    dfa_w = pd.concat([dfa_w, df_result1.groupby(['date'])[['weighted_y']].sum()])
dfa = dfa.reset_index(drop=False)
dfa_w = dfa_w.reset_index(drop=False)
result_all.to_csv(os.path.join('results', 'result_all.csv'), index=False)
# plot return
# rates = [0, 2, 3, 5, 10]
rates = [10]
for rate in rates:
    dfa['y{}'.format(rate)] = dfa.y + 1 - rate / 1000
    dfa['cum{}'.format(rate)] = dfa['y{}'.format(rate)].cumprod()
    dfa_w['y{}'.format(rate)] = dfa_w.weighted_y + 1 - rate / 1000
    dfa_w['cum{}'.format(rate)] = dfa_w['y{}'.format(rate)].cumprod()
    x = dfa.date.values
    x = [str(x)[:10] for x in x]
    y1 = dfa.y.values - rate / 1000
    y2 = dfa['cum{}'.format(rate)].values
    y3 = dfa_w.weighted_y.values - rate / 1000
    y4 = dfa_w['cum{}'.format(rate)].values
    label_x = 'date'
    # label_y1 = '周收益率'
    # label_y2 = '累计收益率'
    color_y1 = '#2A9CAD'  # blue
    color_y2 = "#FAB03D"  # orange
    color_y3 = '#3dfa6c'  # green
    color_y4 = '#fa3d3d'  # red
    title = 'montly return and accumulate trading fee ‰{}'.format(rate)

    fig, ax1 = plt.subplots(figsize=(10, 7), dpi=250)
    plt.xticks(rotation=60)
    ax2 = ax1.twinx()  # 做镜像处理

    lns1 = ax1.bar(x=x, height=y1, color=color_y1, alpha=0.7)
    lns2 = ax2.plot(x, y2, color=color_y2, ms=10)
    lns3 = ax1.bar(x=x, height=y3, color=color_y3, alpha=0.7)
    lns4 = ax2.plot(x, y4, color=color_y4, ms=10)

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
# predict using the rest of the dates
tmp_result_all = pd.DataFrame()
for tmp_date in df_dates[df_dates.trade_date > end_dates.iloc[-1]].trade_date:
    print('预测时间：', tmp_date)
    df_test = df[df.date == tmp_date].reset_index(drop=True)

    df_test[f_x] = mevtransformer.transform(df_test[f_x])
    df_test[f_x] = scaler.transform(df_test[f_x])

    # regression
    y_pred = model.predict(df_test[f_x]).flatten()
    # classification
    y_pred_class = classification_model.predict(df_test[f_x]).flatten()

    # result
    df_result = df_test[['index_code', 'date']].copy()
    df_result['y'] = df_test['target']
    df_result['y_pred'] = y_pred
    df_result['y_pred_rank'] = df_result.y_pred.rank(ascending=False)
    df_result['class_label'] = df_result['y_pred_rank'].transform(
        lambda x: pd.qcut(x, 3, labels=range(1, 4), duplicates='drop'))
    df_result['y_pred_class'] = y_pred_class

    df_result1 = df_result[df_result.y_pred_class == 1]
    df_result1 = df_result1[df_result1.class_label == 1]
    df_result1.to_csv(os.path.join('tmp_results', str(tmp_date) + '.csv'), index=False)
    tmp_result_all = pd.concat([tmp_result_all, df_result1])
    tmp_result_all.to_csv(os.path.join('tmp_results', 'result_all.csv'), index=False)

