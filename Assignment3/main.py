from keras.layers import Dense, Convolution2D, LeakyReLU, BatchNormalization, Flatten, Dropout, MaxPool2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras import backend

import utilities as utils

if __name__ == "__main__":
    backend.clear_session()
    models = []

    # Example model
    model_ex = Sequential(name="Example_Model")
    model_ex.add(Dense(100, activation='relu', input_shape=(9216,)))
    model_ex.add(Dense(30))
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model_ex.compile(loss='mean_squared_error', optimizer=sgd)
    # model_ex.summary()
    models.append(model_ex)

    # CNN model
    model_cnn = Sequential(name="CNN_Model")
    model_cnn.add(Convolution2D(32, (3, 3), padding='same', use_bias=False, input_shape=(96, 96, 1)))
    model_cnn.add(LeakyReLU(alpha=0.1))
    model_cnn.add(BatchNormalization())
    model_cnn.add(Convolution2D(32, (3, 3), padding='same', use_bias=False))
    model_cnn.add(LeakyReLU(alpha=0.1))
    model_cnn.add(BatchNormalization())
    model_cnn.add(MaxPool2D(pool_size=(2, 2)))
    model_cnn.add(Convolution2D(64, (3, 3), padding='same', use_bias=False))
    model_cnn.add(LeakyReLU(alpha=0.1))
    model_cnn.add(BatchNormalization())
    model_cnn.add(Convolution2D(64, (3, 3), padding='same', use_bias=False))
    model_cnn.add(LeakyReLU(alpha=0.1))
    model_cnn.add(BatchNormalization())
    model_cnn.add(MaxPool2D(pool_size=(2, 2)))
    model_cnn.add(Convolution2D(96, (3, 3), padding='same', use_bias=False))
    model_cnn.add(LeakyReLU(alpha=0.1))
    model_cnn.add(BatchNormalization())
    model_cnn.add(Convolution2D(96, (3, 3), padding='same', use_bias=False))
    model_cnn.add(LeakyReLU(alpha=0.1))
    model_cnn.add(BatchNormalization())
    model_cnn.add(MaxPool2D(pool_size=(2, 2)))
    model_cnn.add(Convolution2D(128, (3, 3), padding='same', use_bias=False))
    model_cnn.add(LeakyReLU(alpha=0.1))
    model_cnn.add(BatchNormalization())
    model_cnn.add(Convolution2D(128, (3, 3), padding='same', use_bias=False))
    model_cnn.add(LeakyReLU(alpha=0.1))
    model_cnn.add(BatchNormalization())
    model_cnn.add(MaxPool2D(pool_size=(2, 2)))
    model_cnn.add(Convolution2D(256, (3, 3), padding='same', use_bias=False))
    model_cnn.add(LeakyReLU(alpha=0.1))
    model_cnn.add(BatchNormalization())
    model_cnn.add(Convolution2D(256, (3, 3), padding='same', use_bias=False))
    model_cnn.add(LeakyReLU(alpha=0.1))
    model_cnn.add(BatchNormalization())
    model_cnn.add(MaxPool2D(pool_size=(2, 2)))
    model_cnn.add(Convolution2D(512, (3, 3), padding='same', use_bias=False))
    model_cnn.add(LeakyReLU(alpha=0.1))
    model_cnn.add(BatchNormalization())
    model_cnn.add(Convolution2D(512, (3, 3), padding='same', use_bias=False))
    model_cnn.add(LeakyReLU(alpha=0.1))
    model_cnn.add(BatchNormalization())
    model_cnn.add(Flatten())
    model_cnn.add(Dense(512, activation='relu'))
    model_cnn.add(Dropout(0.1))
    model_cnn.add(Dense(30))
    model_cnn.compile(optimizer='adam',
                            loss='mean_squared_error',
                            metrics=['mae'])
    # model_cnn.summary()
    models.append(model_cnn)

    # CNN model
    model_small_cnn = Sequential(name="Small_CNN_Model")
    model_small_cnn.add(Convolution2D(32, (3, 3), padding='same', use_bias=False, input_shape=(96, 96, 1)))
    model_small_cnn.add(LeakyReLU(alpha=0.1))
    model_small_cnn.add(BatchNormalization())
    model_small_cnn.add(Convolution2D(32, (3, 3), padding='same', use_bias=False))
    model_small_cnn.add(LeakyReLU(alpha=0.1))
    model_small_cnn.add(BatchNormalization())
    model_small_cnn.add(MaxPool2D(pool_size=(2, 2)))
    model_small_cnn.add(Convolution2D(64, (3, 3), padding='same', use_bias=False))
    model_small_cnn.add(LeakyReLU(alpha=0.1))
    model_small_cnn.add(BatchNormalization())
    model_small_cnn.add(Convolution2D(64, (3, 3), padding='same', use_bias=False))
    model_small_cnn.add(LeakyReLU(alpha=0.1))
    model_small_cnn.add(BatchNormalization())
    model_small_cnn.add(MaxPool2D(pool_size=(2, 2)))
    model_small_cnn.add(Convolution2D(128, (3, 3), padding='same', use_bias=False))
    model_small_cnn.add(LeakyReLU(alpha=0.1))
    model_small_cnn.add(BatchNormalization())
    model_small_cnn.add(Convolution2D(128, (3, 3), padding='same', use_bias=False))
    model_small_cnn.add(LeakyReLU(alpha=0.1))
    model_small_cnn.add(BatchNormalization())
    model_small_cnn.add(Flatten())
    model_small_cnn.add(Dense(128, activation='relu'))
    model_small_cnn.add(Dropout(0.1))
    model_small_cnn.add(Dense(30))
    model_small_cnn.compile(optimizer='adam',
                            loss='mean_squared_error',
                            metrics=['mae'])
    # model_small_cnn.summary()
    models.append(model_small_cnn)
    epochs = 10
    batch_size = 128
    for model in models:
        model, history = utils.test_model(model, epochs)
        model, history = utils.load_model(model.name)
        utils.plot_loss(model, history)
        utils.show_examples(model)
        # utils.plot_model_to_file(model)
