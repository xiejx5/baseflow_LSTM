import os
import numpy as np
import pickle
import tensorflow as tf
from keras import backend as K
from cond_lstm import (get_info, load_data, train, create_model)


def train_work(eco):
    if os.path.exists('..\\Data\\Model\\' + eco + '.h5'):
        return None

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        K.set_session(sess)

        # train index and data prepare
        gages, factor, flow = get_info(1979, 1, eco)
        train_index = np.arange(flow.shape[0])

        # load train data
        (cond, X, y), (cond_mean, X_mean, y_mean), (cond_std, X_std, y_std) = load_data(
            train_index, gages, factor, flow)

        # model construct and train
        model = create_model()
        C = [cond, X]
        h = train(model, C, y, verbose=0)

        # save model
        model.save('..\\Data\\Model\\' + eco + '.h5')

    # Saving the objects:
    norm = [cond_mean, cond_std, X_mean, X_std, y_mean, y_std]
    with open('..\\Data\\Model\\' + eco + '.pickle', 'wb') as f:
        pickle.dump(norm, f)

    return h
