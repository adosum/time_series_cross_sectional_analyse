import talib as ta
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import linear_model

# macd 30 days
# dea 10 days
# dif

fastperiod = 10
slowperiod = 30
signalperiod = 9


def macd(x, fastperiod=10, slowperiod=30, signalperiod=9):
    macd, dea, dif = ta.MACD(x,
                             fastperiod=fastperiod,
                             slowperiod=slowperiod,
                             signalperiod=signalperiod)
    return macd


def dea(x, fastperiod=10, slowperiod=30, signalperiod=9):
    macd, dea, dif = ta.MACD(x,
                             fastperiod=fastperiod,
                             slowperiod=slowperiod,
                             signalperiod=signalperiod)
    return dea


def dif(x, fastperiod=10, slowperiod=30, signalperiod=9):
    macd, dea, dif = ta.MACD(x,
                             fastperiod=fastperiod,
                             slowperiod=slowperiod,
                             signalperiod=signalperiod)
    return dif


def rsi(x, rsiperiod=20):
    result = ta.RSI(x, timeperiod=rsiperiod)
    return result


# psy 20日
def psy(x, period=20):
    difference = x[1:].values - x[:-1].values
    difference_dir = np.where(difference > 0, 1, 0)
    p = np.zeros((len(x),))
    p[:period] *= np.nan
    for i in range(period, len(x)):
        p[i] = (difference_dir[i - period + 1:i + 1].sum()) / period
    return pd.Series(p * 100, index=x.index)


def bias(x, biasperiod=20):
    result = (x - x.rolling(biasperiod, min_periods=1).mean()) / x.rolling(biasperiod, min_periods=1).mean() * 100
    result = round(result, 2)
    return result


class MedianExtremeValueTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dm_s = None
        self.dm1_s = None

    def fit(self, df_name):
        self.dm_s = df_name.median()
        self.dm1_s = df_name.apply(lambda x: x - self.dm_s[x.name]).abs().median()
        return self

    def scaller(self, x):
        self.di_max = self.dm_s[x.name] + 5 * self.dm1_s[x.name]
        self.di_min = self.dm_s[x.name] - 5 * self.dm1_s[x.name]
        x = x.apply(lambda v: self.di_min if v < self.di_min else v)
        x = x.apply(lambda v: self.di_max if v > self.di_max else v)
        return x

    def transform(self, df_name):
        df_name = df_name.apply(self.scaller)
        return df_name


class GroupValueFiller(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.df_mean_industry = None

    def fit(self, df_name, f_x, group_name='industry'):
        self.df_mean_industry = df_name.groupby(group_name)[f_x].mean()
        self.df_mean_industry = self.df_mean_industry.fillna(self.df_mean_industry.mean())
        self.df_mean_industry.columns = [x + '_mean' for x in self.df_mean_industry.columns]
        self.df_mean_industry = self.df_mean_industry.reset_index()
        return self

    def transform(self, df_name, f_x, group_name='industry'):
        df_name_mean = df_name.merge(self.df_mean_industry, on=group_name, how='left')
        df_name_mean[f_x] = df_name_mean[f_x].apply(lambda x: x.fillna(df_name_mean[x.name + '_mean']))
        df_name = df_name_mean[df_name.columns]
        return df_name


class IndustryNeutral(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.models = {}

    def regr_fit(self, X, y):
        self.models[y.name] = linear_model.LinearRegression().fit(X, y)
        return

    def regr_pred(self, X, y):
        pred = self.models[y.name].predict(X)
        return y - pred

    def fit(self, df_name, f_x, f_idst):
        X = df_name[['total_mv_log'] + f_idst]
        df_name[f_x].apply(lambda y: self.regr_fit(X, y))
        return self

    def transform(self, df_name, f_x, f_idst):
        X = df_name[['total_mv_log'] + f_idst]
        df_name[f_x] = df_name[f_x].apply(lambda y: self.regr_pred(X, y))
        return df_name