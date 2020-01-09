import codecs
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from keras.models import model_from_json
from keras.utils.vis_utils import plot_model
from sklearn.utils import shuffle

FTRAIN = 'data/training.csv'
FTEST = 'data/test.csv'
FLOOKUP = 'data/IdLookupTable.csv'


def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    f_name = FTEST if test else FTRAIN
    df = pd.read_csv(f_name)
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    if cols:
        df = df[list(cols) + ['Image']]
    df = df.dropna()
    x = np.vstack(df['Image'].values) / 255.
    x = x.astype(np.float32)
    if not test:
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48
        x, y = shuffle(x, y, random_state=42)
        y = y.astype(np.float32)
    else:
        y = None
    return x, y


def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)


def plot_loss(model, history):
    loss = history['loss']
    val_loss = history['val_loss']
    plt.plot(loss, linewidth=3, label='train')
    plt.plot(val_loss, linewidth=3, label='valid')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.title(model.name)
    final_score = "Loss: " + str(round(history['loss'][-1], 8)) + \
                  "\nVal_Loss: " + str(round(history['val_loss'][-1], 8))
    adjust_text([plt.text(len(history['loss']), np.mean(history['loss']), final_score, size='large')], history["loss"],
                np.arange(len(history["loss"])))
    plt.show()


def show_examples(model):
    x_test, _ = load(test=True)
    if "CNN_Model" in model.name:
        x_test = x_test.reshape((-1, 96, 96, 1))
    y_pred = model.predict(x_test)
    fig = plt.figure(figsize=(6, 6))
    fig.suptitle(model.name)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(x_test[i], y_pred[i], ax)

    plt.show()


def test_model(model, epochs=10, batch_size=32, save=True):
    # x, y = load(test=False)
    x = np.load('data/x.npy')
    y = np.load('data/y.npy')
    if "CNN_Model" in model.name:
        x = x.reshape((-1, 96, 96, 1))

    history = model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.2).history
    if save:
        save_model(model, history)
    return model, history


def save_model(model, history):
    model_json = model.to_json()
    with open("models/" + model.name + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("models/" + model.name + ".h5")
    with open("models/" + model.name + "_history.json", "w") as file:
        json.dump(history, file)
    print("Saved model to disk")


def load_model(model_name):
    json_file = open("models/" + model_name + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("models/" + model_name + ".h5")
    with codecs.open("models/" + model_name + "_history.json", "r", "utf-8") as json_file:
        history = json.loads(json_file.read())
    print("Loaded model from disk")
    return loaded_model, history


def plot_model_to_file(model):
    plot_model(model, to_file="images/" + model.name + "_plot.png", show_shapes=True,
               show_layer_names=False)
