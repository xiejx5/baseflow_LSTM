# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
from keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.compat.v1.keras.layers import LSTM
from _const import (TIME_STEPS, BATCH_SIZE, NUM_CELLS, DROP_P,
                    EPOCHS, COND_DIM, INPUT_DIM, OUTPUT_DIM)


def load_data(train_index, gages, factor, flow, Normalized=None):
    # generate cond, cond_mean, cond_std
    from_gage = np.digitize(train_index, gages['t'])
    train_gages = np.unique(from_gage)
    if Normalized is None:
        # cond_mean = factor.iloc[train_gages].mean(axis=0)
        # cond_std = factor.iloc[train_gages].std(axis=0)
        cond_mean = factor.iloc[train_gages].min(axis=0)
        cond_std = factor.iloc[train_gages].max(axis=0) - cond_mean
        # is_percent = (cond_mean.index != 'ELEV') & (cond_mean.index != 'PERM')
        # cond_mean[is_percent] = 0
        # cond_std[is_percent] = 100
    else:
        cond_mean = Normalized[0]
        cond_std = Normalized[1]
    cond = factor.iloc[from_gage]
    cond = (cond - cond_mean) / cond_std

    X = np.zeros((train_index.shape[0], TIME_STEPS, INPUT_DIM))
    for i in train_gages:
        X_index = np.where(from_gage == i)[0]
        sample_i = train_index[X_index]
        df = pd.read_csv('..\\Data\\Basin_Time\\' +
                         gages['STAID'].iloc[i] + '.csv')
        df_index = flow['Delta'].iloc[sample_i]
        substract = np.tile(np.arange(0, TIME_STEPS)[::-1], df_index.shape[0])
        df_index = np.repeat(df_index, TIME_STEPS)
        df_index = df_index - substract
        X[X_index] = np.array(df.iloc[df_index]).reshape(X[X_index].shape)
    if Normalized is None:
        X_mean = X.reshape(X.shape[0] * X.shape[1], X.shape[2]).mean(axis=0)
        X_std = X.reshape(X.shape[0] * X.shape[1], X.shape[2]).std(axis=0)
    else:
        X_mean = Normalized[2]
        X_std = Normalized[3]
    X = (X - X_mean) / X_std

    y = flow['B'].iloc[train_index]
    if Normalized is None:
        # y_mean = y.mean(axis=0)
        # y_std = y.std(axis=0)
        y_mean = 0
        y_std = y.max(axis=0)
    else:
        y_mean = Normalized[4]
        y_std = Normalized[5]
    y = (y - y_mean) / y_std

    return (cond, X, y), (cond_mean, X_mean, y_mean), (cond_std, X_std, y_std)


def get_info(start_year, start_month, eco=None):
    flow = pd.DataFrame()
    gages = pd.read_excel('..\\Data\\Gages.xlsx', dtype={'STAID': str})
    gages['num_months'] = -1
    factor = pd.read_excel('..\\Data\\Factor_Cond.xlsx')
    if eco is not None:
        factor = factor[gages['ECO'] == eco]
        gages = gages[gages['ECO'] == eco]
    for index, g in gages.iterrows():
        df = pd.read_csv('..\\Data\\Basin_MonthFlow\\' + g['STAID'] + '.csv')
        reserve = np.where((df['Y'] == start_year + (TIME_STEPS - 1) // 12) &
                           (df['M'] >= start_month + (TIME_STEPS - 1) % 12) |
                           (df['Y'] >= start_year + 1 + (TIME_STEPS - 1) // 12))
        if reserve[0].shape[0]:
            flow_start = reserve[0][0]
            flow = pd.concat([flow, df.iloc[flow_start:]],
                             axis=0, ignore_index=True)
            gages.loc[index, 'num_months'] = df.iloc[flow_start:].shape[0]

    flow['Delta'] = (flow['Y'] - start_year) * 12 + flow['M'] - 1
    factor = factor[gages['num_months'] != -1]
    # factor = factor.iloc[:, factor.columns.get_loc('ELEV'):factor.shape[1]]
    gages = gages[gages['num_months'] != -1]
    gages['t'] = np.cumsum(gages['num_months'])
    gages['s'] = gages['t'].shift(1, fill_value=0)
    flow['C'] = np.repeat(
        np.arange(0, gages.shape[0], dtype=int), gages['t'] - gages['s'])

    return gages, factor, flow


def NSE(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def create_model():
    # construct model
    feature_input = Input(shape=(COND_DIM,))
    h0 = Dense(NUM_CELLS, activation='relu')(feature_input)
    h0 = Dense(NUM_CELLS)(h0)
    c0 = Dense(NUM_CELLS, activation='relu')(feature_input)
    c0 = Dense(NUM_CELLS)(c0)
    series_input = Input(shape=(TIME_STEPS, INPUT_DIM))
    lstm = LSTM(NUM_CELLS, recurrent_initializer='Orthogonal',
                kernel_initializer='Orthogonal',
                implementation=2,
                recurrent_dropout=DROP_P)(series_input, initial_state=[h0, c0])
    out = Dense(OUTPUT_DIM, activation='relu')(lstm)
    model = Model(inputs=[feature_input, series_input], outputs=out)
    model.compile(loss='mse', optimizer='adam', metrics=[NSE])

    return model


def create_model_N():
    # construct model
    model_N = Sequential()
    model_N.add(LSTM(NUM_CELLS,  recurrent_initializer='Orthogonal',
                     kernel_initializer='Orthogonal',
                     implementation=2,
                     recurrent_dropout=DROP_P,
                     input_shape=(TIME_STEPS, INPUT_DIM)))
    model_N.add(Dense(OUTPUT_DIM, activation='relu'))
    model_N.compile(optimizer='adam', loss='mse', metrics=[NSE])

    return model_N


class Pred_History(Callback):
    def __init__(self, model, X_test, return_epoch):
        self.model = model
        self.pred_his = []
        self.X_test = X_test
        self.return_epoch = return_epoch

    def on_epoch_end(self, epoch, logs={}):
        if isinstance(self.return_epoch, int):
            if epoch % self.return_epoch == 0:
                self.pred_his.append(self.model.predict(
                    self.X_test, batch_size=self.X_test.shape[0]).squeeze())
        else:
            if epoch in self.return_epoch:
                self.pred_his.append(self.model.predict(
                    self.X_test, batch_size=self.X_test.shape[0]).squeeze())


def train_and_test(model, C, y, C_test, y_test, return_epoch=None, **kwargs):
    if return_epoch is not None:
        epoch_save = Pred_History(model, C_test, return_epoch)
        epoch_save.pred_his.append(
            model.predict(C_test, batch_size=BATCH_SIZE).squeeze())
        h = model.fit(C, y, epochs=EPOCHS, shuffle=True,
                      batch_size=BATCH_SIZE, **kwargs,
                      validation_data=(C_test, y_test),
                      callbacks=[epoch_save])
        return h, epoch_save.pred_his

    h = model.fit(C, y, epochs=EPOCHS, shuffle=True,
                  batch_size=BATCH_SIZE, **kwargs,
                  validation_data=(C_test, y_test))
    return h


def train(model, C, y, return_epoch=None, **kwargs):
    h = model.fit(C, y, epochs=EPOCHS, shuffle=True,
                  batch_size=BATCH_SIZE, **kwargs)
    return h

    # y_pred = model.predict(C_test, batch_size=BATCH_SIZE).squeeze()
    # nse1 = 1 - np.sum(np.power(y_pred - y_test, 2)) / \
    #     np.sum(np.power(y_test - y_test.mean(axis=0), 2))
