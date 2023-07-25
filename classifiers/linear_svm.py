from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras import initializers


def svm(classes, input_shape, weight_decay):
    """
    create a simple linear SVM classifier.
    :param classes: output num classes
    :param input_shape: input image shape
    :return:
    """
    model = Sequential()
    model.add(Dense(classes,
                    input_shape=input_shape,
                    kernel_regularizer=regularizers.l2(weight_decay),
                    kernel_initializer=initializers.random_normal(stddev=1e-3),
                    use_bias=True,
                    ))
    return model
