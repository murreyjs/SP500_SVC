import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def main():
    """
    Small program that uses a Support Vector Classification model on
    SPY daily stats (Open, Close, High, Low) in order to predict
    if the Closing price will increase the next day
    """
    df = get_data()
    x = get_features(df)
    y = get_targets(df)
    x_train, y_train, x_test, y_test = split_data_sets(df, x, y)
    cls = SVC().fit(x_train, y_train)
    y_predict = cls.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_predict)
    print(accuracy_test)
    all_preds = cls.predict(x)
    plot_cumulative_returns(cls, all_preds)



def get_data():
    """
    :return: data frame of SPY daily stats including additionally calculated features
    """
    df = pd.read_csv("./spy_daily.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low
    df['Open/Close'] = df.Open / df.Close
    df['High/Low'] = df.High / df.Low
    df['High-Open'] = df.High - df.Open
    df['Open-Low'] = df.Open - df.Low
    df['High-Close'] = df.High - df.Close
    df['Close-Low'] = df.Close - df.Low
    return df


def get_features(df):
    """
    :param df: data frame of SPY daily stats
    :return: relevant columns used as features in model
    """
    return df[['Open-Close', 'High-Low', 'High-Open', 'Open-Low', 'High-Close', 'Close-Low']]


def get_targets(df):
    """
    :param df: data frame of SPY daily stats
    :return: column used as target in model
    """
    return np.where(df['Close'].shift(-1) > df['Close'], 1, 0)


def split_data_sets(df, x, y, split_percentage=0.8):
    """
    :param df: data frame of SPY daily stats
    :param x: feature columns
    :param y: target column
    :param split_percentage: percentage of data allocated to train the model
    :return: train features, train targets, test features, test targets
    """
    split = int(split_percentage*len(df))
    x_train = x[:split]
    x_test = x[split:]

    y_train = y[:split]
    y_test = y[split:]
    return x_train, y_train, x_test, y_test


def plot_cumulative_returns(pred, df):
    """
    :param cls: classifier model
    :param df: data frame of SPY daily stats
    :param x: prepared features of the data frame
    :return:
    """
    df['Predicted_Signal'] = pred
    df['Returns'] = df.Close.pct_change()
    df['Strategy_Returns'] = df.Returns * df.Predicted_Signal.shift(1)
    df['cumulative_returns'] = (df.Strategy_Returns + 1).cumprod()

    plt.title("Cumulative Returns Plot", fontsize=16)
    plt.ylabel("Cumulative Returns")
    plt.xlabel("Date")
    df['cumulative_returns'].plot(figsize=(15, 7), color='g')
    plt.show()
