import os
import cv2
import random

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPool2D, Dense, Activation, Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils


ROWS = 64
COLS = 64
CHANNELS = 3
TRAIN_DIR = './data/train/'
TEST_DIR = './data/test/'


def read_image(file_path):
    """
    处理图片，对图片进行缩放处理
    :param file_path:
    :return:
    """
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)  # 读出来的数据格式是0-255的BGR格式
    # interpolation是定义插值方法
    newImage = cv2.resize(image, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    return newImage


def prep_data(file_paths):
    """
    将所有的图片进行缩放处理，并将二进制字符保存到numpy数组中
    :param file_paths:
    :return:
    """
    count = len(file_paths)
    data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.float32)
    for i, image_file in enumerate(file_paths):
        image = read_image(image_file)
        data[i] = image

    return data


def loadDataSet():
    """
    读取文件，并且将数据存入到numpy数组中，返回测试集，训练集和训练标签，并对返回的数据做归一化处理
    :return:
    """
    train_images = [TRAIN_DIR + filename for filename in os.listdir(TRAIN_DIR)]
    # train_dogs = [TRAIN_DIR + filename for filename in os.listdir(TRAIN_DIR) if 'dog' in filename][:1000]
    # train_cats = [TRAIN_DIR + filename for filename in os.listdir(TRAIN_DIR) if 'cat' in filename][:1000]
    # train_images = train_dogs + train_cats
    test_images = [TEST_DIR + filename for filename in os.listdir(TEST_DIR)]
    random.shuffle(train_images)
    labels = np.array([[1] if 'dog' in filename else [0] for filename in train_images])

    train_data = prep_data(train_images) / 255
    test_data = prep_data(test_images) / 255

    return train_data, labels, test_data


optimizer = RMSprop(lr=1e-4)
objective = "binary_crossentropy"


def catDogModel():
    model = Sequential()
    # 两层卷积层加一层池化层
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(ROWS, COLS, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])

    return model


nb_epoch = 10000
batch_size = 128


class LossHistory(Callback):
    """
    该类用来作为回调函数输出损失日志
    """

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))


early_stopping = EarlyStopping(monitor="val_loss", patience=3, verbose=1, mode='auto')


def run_catdog():
    model = catDogModel()
    train, labels, test = loadDataSet()
    history = LossHistory()
    model.fit(train, labels, batch_size=batch_size, epochs=nb_epoch, validation_split=0.25, verbose=0,
              shuffle=True, callbacks=[history, early_stopping])
    model.save('./model')

    predictions = model.predict(test, verbose=0)

    predict = pd.DataFrame({"id": range(len(test)), 'label': predictions})

    predict.to_csv('./data/predictions.csv', index=False)
    return predictions, history


predictions, history = run_catdog()
print(predictions)