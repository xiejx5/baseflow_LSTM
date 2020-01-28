import tensorflow as tf
from keras import backend as K
from cond_lstm import (load_data, create_model,
                       create_model_N, train_and_test)


def parallel_val(args):
    (i, (train_index, test_index)), (gages, factor, flow) = args

    # init session
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        K.set_session(sess)

        # load train data
        (cond, X, y), (cond_mean, X_mean, y_mean), (cond_std, X_std, y_std) = load_data(
            train_index, gages, factor, flow)

        # load test data
        cond_test, X_test, y_test = load_data(
            test_index, gages, factor, flow,
            Normalized=(cond_mean, cond_std, X_mean, X_std, y_mean, y_std))[0]

        # model construct train test
        model = create_model()
        C = [cond, X]
        C_test = [cond_test, X_test]
        h = train_and_test(model, C, y, C_test, y_test, verbose=0)

        # model_N construct train test
        model_N = create_model_N()
        h_N = train_and_test(model_N, X, y, X_test, y_test, verbose=0)

    # output
    print(i)
    return (i, h.history['val_NSE'][-1], h_N.history['val_NSE'][-1])
