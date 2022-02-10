import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# 时间格式转化
def parser(x):
    #return pd.to_datetime(x, format='%m/%d/%Y %H:%M:%S')
    return pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')


# 将数据转换为监督学习型数据，删掉NaN值
def timeseries_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # 获取特征值数量n_vars
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    # print(df)

    cols, names = list(), list()

    # input sequence(t-n, ... , t-1)
    # 创建n个v(t-1)作为列名
    for i in range(n_in, 0, -1):
        # 向列表cols中添加1个df.shift(1)的数据
        cols.append(df.shift(i))
        # print(cols)
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    # output sequence (t, t+1, ... , t+n)
    for i in range(0, n_out):
        # 向列表cols中添加1个df.shift(-1)的数据
        cols.append(df.shift(-i))
        # print(cols)
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # print(cols)

    # 将列表中的两个张量按照列拼接起来，list(v1, v2) -> [v1, v2], 其中v1向下移动了1行，此时v1, v2是监督学习型数据
    agg = pd.concat(cols, axis=1)
    # print(agg)

    # 重定义列名
    agg.columns = names
    # print(agg)

    # 删除空值
    if dropnan:
        agg.dropna(inplace=True)

    return agg


# 将数据缩放到[-1,1]之间的数
def scale(dataset):
    # 创建1个缩放器
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(dataset)
    # print(dataset)

    # 缩放
    dataset_scaled = scaler.transform(dataset)
    # print(dataset_scaled)

    return scaler, dataset_scaled


# 逆缩放
def invert_scale(scaler, yhat, y):
    # 逆缩放输入形状为(n,2), 输出形状为(n,2)
    inverted_yhat = scaler.inverse_transform(yhat)
    inverted_y = scaler.inverse_transform(y)

    return inverted_yhat, inverted_y


# 数据拆分：拆分为训练集，测试集
def split_dataset(dataset, seq_len=100):
    # 数据集长度
    length = len(dataset)

    # 初始训练集索引为空列表
    index_train = []
    # 初始测试集索引为空列表
    index_test = []

    for i in range(length):
        if (i % seq_len < seq_len - 30):
            index_train.append(i)
        else:
            index_test.append(i)

    # 前70%个作为训练集
    # train = [dataset[index] for index in index_train]
    train = np.array(dataset)[index_train]
    # 后30%个作为测试集
    # test = [dataset[index] for index in index_test]
    test = np.array(dataset)[index_test]

    return train, test


def detect_elec():
    # load model from single file
    elec_model = tf.keras.models.load_model('./models/lstm_model_elec.h5')
    # 定义特征个数(用变量表示特征，标签)
    n_features = 41

    # 加载数据
    df = pd.read_csv('./dataset/电源分系统遥测.csv', dtype=object, header=0, index_col=0, parse_dates=[0], squeeze=True, date_parser=parser)
    raw_values = df.values
    # 拆分训练集和测试集
    x1, x2 = split_dataset(raw_values, seq_len=100)
    # 将所有数据缩放到[-1, 1]之间
    scaler, scaled_values = scale(x1)

    # 加载数据
    df_elec = pd.read_csv('./dataset/电源分系统遥测-测试.csv', dtype=object, header=0, index_col=0, parse_dates=[0], squeeze=True, date_parser=parser)
    elec_values = df_elec.values
    # 将所有数据缩放到[-1, 1]之间
    elec_scaled_values = scaler.transform(elec_values)
    # 将数据转换为监督学习型数据
    elec_supervised = timeseries_to_supervised(elec_scaled_values, 1, 1)
    elec_supervised_values = elec_supervised.values
    # 目标值和特征值
    elec_X, elec_y = elec_supervised_values[:, :n_features], elec_supervised_values[:, n_features:]
    # 将输入数据转换成3维张量[samples, timesteps, features], [1次批量n条数据， 每条数据1个步长，41个特征值]
    elec_X = elec_X.reshape((elec_X.shape[0], 1, n_features))
    # 使用训练好的模型网络进行预测
    elec_yhat = elec_model.predict(elec_X, batch_size=1)
    elec_yhat = elec_yhat.reshape(elec_yhat.shape[0], elec_yhat.shape[1])
    # 对替换后的inv_yhat预测数据进行逆缩放
    elec_inv_yhat = scaler.inverse_transform(elec_yhat)
    # 对重构后数据进行逆缩放
    elec_inv_y = scaler.inverse_transform(elec_y)

    flag = False
    #error_count = 0
    for i in range(0, 2456, 3):
        if (
                (abs(elec_inv_y[i, 0] - elec_inv_yhat[i, 0]) > 2) &
                (abs(elec_inv_y[i + 1, 0] - elec_inv_yhat[i + 1, 0]) > 2) &
                (abs(elec_inv_y[i + 2, 0] - elec_inv_yhat[i + 2, 0]) > 2)
        ):
            #error_count += 1
            #print("电源分系统故障:", error_count * 50, i)
            print("电源分系统故障:", i)
            flag = True
            break

    return flag

def detect_temp():
    # load model from single file
    temp_model = tf.keras.models.load_model('./models/lstm_model_temp.h5')
    # 定义特征个数(用变量表示特征，标签)
    n_features = 59

    # 加载数据
    df = pd.read_csv('./dataset/热控分系统遥测.csv', dtype=object, header=0, index_col=0, parse_dates=[0], squeeze=True, date_parser=parser)
    raw_values = df.values
    # 拆分训练集和测试集
    x1, x2 = split_dataset(raw_values, seq_len=100)
    # 将所有数据缩放到[-1, 1]之间
    scaler, scaled_values = scale(x1)

    # 加载数据
    df_temp = pd.read_csv('./dataset/热控分系统遥测-测试.csv', dtype=object, header=0, index_col=0, parse_dates=[0], squeeze=True, date_parser=parser)
    temp_values = df_temp.values
    # 将所有数据缩放到[-1, 1]之间
    temp_scaled_values = scaler.transform(temp_values)
    # 将数据转换为监督学习型数据
    temp_supervised = timeseries_to_supervised(temp_scaled_values, 2, 1)
    temp_supervised_values = temp_supervised.values
    # 目标值和特征值
    temp_X, temp_y = temp_supervised_values[:, :2 * n_features], temp_supervised_values[:, 2 * n_features:]
    # 将输入数据转换成3维张量[samples, timesteps, features], [1次批量n条数据， 每条数据2个步长，59个特征值]
    temp_X = temp_X.reshape((temp_X.shape[0], 2, n_features))
    # 使用训练好的模型网络进行预测
    temp_yhat = temp_model.predict(temp_X, batch_size=2)
    temp_yhat = temp_yhat.reshape(temp_yhat.shape[0], temp_yhat.shape[1])
    # 对替换后的inv_yhat预测数据进行逆缩放
    temp_inv_yhat = scaler.inverse_transform(temp_yhat)
    # 对重构后数据进行逆缩放
    temp_inv_y = scaler.inverse_transform(temp_y)

    flag = False
    #error_count = 0
    for i in range(0, 140, 3):
        if (
                (abs(temp_inv_y[i, 0] - temp_inv_yhat[i, 0]) > 5) &
                (abs(temp_inv_y[i + 1, 0] - temp_inv_yhat[i + 1, 0]) > 5) &
                (abs(temp_inv_y[i + 2, 0] - temp_inv_yhat[i + 2, 0]) > 5)
        ):
            #error_count += 1
            #print("热控分系统故障:", error_count * 20, i)
            print("热控分系统故障:", i)
            flag = True
            break


    return flag

def detect_adcs():
    # load model from single file
    adcs_model = tf.keras.models.load_model('./models/lstm_model_adcs.h5')
    # 定义特征个数(用变量表示特征，标签)
    n_features = 14

    # 加载数据
    df = pd.read_csv('./dataset/姿控分系统遥测.csv', dtype=object, header=0, index_col=0, parse_dates=[0], squeeze=True, date_parser=parser)
    raw_values = df.values
    # 拆分训练集和测试集
    x1, x2 = split_dataset(raw_values, seq_len=100)
    # 将所有数据缩放到[-1, 1]之间
    scaler, scaled_values = scale(x1)

    # 加载数据
    df_adcs = pd.read_csv('./dataset/姿控分系统遥测-测试.csv', dtype=object, header=0, index_col=0, parse_dates=[0], squeeze=True, date_parser=parser)
    adcs_values = df_adcs.values
    # 将所有数据缩放到[-1, 1]之间
    adcs_scaled_values = scaler.transform(adcs_values)
    # 将数据转换为监督学习型数据
    adcs_supervised = timeseries_to_supervised(adcs_scaled_values, 4, 1)
    adcs_supervised_values = adcs_supervised.values
    # 目标值和特征值
    adcs_X, adcs_y = adcs_supervised_values[:, :4 * n_features], adcs_supervised_values[:, 4 * n_features:]
    # 将输入数据转换成3维张量[samples, timesteps, features], [1次批量n条数据， 每条数据4个步长，14个特征值]
    adcs_X = adcs_X.reshape((adcs_X.shape[0], 4, n_features))
    # 使用训练好的模型网络进行预测
    adcs_yhat = adcs_model.predict(adcs_X, batch_size=4)
    adcs_yhat = adcs_yhat.reshape(adcs_yhat.shape[0], adcs_yhat.shape[1])
    # 对替换后的inv_yhat预测数据进行逆缩放
    adcs_inv_yhat = scaler.inverse_transform(adcs_yhat)
    # 对重构后数据进行逆缩放
    adcs_inv_y = scaler.inverse_transform(adcs_y)

    flag = False
    #error_count = 0
    for i in range(0, 2423, 3):
        if (
                (abs(adcs_inv_y[i, 0] - adcs_inv_yhat[i, 0]) > 1) &
                (abs(adcs_inv_y[i + 1, 0] - adcs_inv_yhat[i + 1, 0]) > 1) &
                (abs(adcs_inv_y[i + 2, 0] - adcs_inv_yhat[i + 2, 0]) > 1)
        ):
            #error_count += 1
            #print("姿控分系统故障:", error_count * 50, i)
            print("姿控分系统故障:", i)
            flag = True
            break

    return flag







