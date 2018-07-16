import os
import cv2
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


# 定义全局的常量
ROWS = 64
COLS = 64
CHANNELS = 3
TRAIN_DIR = './data/train/'
TEST_DIR = './data/test/'

tf.logging.set_verbosity(tf.logging.INFO)


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


def catDogModel(features, labels, mode):
    """
    自定义模型, 以VGG-16模型的架构为原型
    :return:
    """
    inputs = tf.reshape(features['x'], [-1, 64, 64, 3])

    # 定义卷积层，总共有12层

    conv1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    conv1_ = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1_, pool_size=(2, 2), strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    conv2_ = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2_, pool_size=(2, 2), strides=2)

    conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    conv3_ = tf.layers.conv2d(inputs=conv3, filters=128, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3_, pool_size=(2, 2), strides=2)

    conv4 = tf.layers.conv2d(inputs=pool3, filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    conv4_ = tf.layers.conv2d(inputs=conv4, filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    pool4 = tf.layers.max_pooling2d(inputs=conv4_, pool_size=(2, 2), strides=2)

    # 定义全连接层，并对于训练时引入dropout
    flatten = tf.reshape(pool4, [-1, 256 * 4 * 4])
    dense1 = tf.layers.dense(inputs=flatten, units=256, activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(inputs=dense1, rate=0.5, training=mode == learn.ModeKeys.TRAIN)

    dense2 = tf.layers.dense(inputs=dropout1, units=256, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(inputs=dense2, rate=0.5, training=mode == learn.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout2, units=1)

    # 定义预测数据
    # if mode == tf.estimator.ModeKeys.PREDICT:
    #     predictions = {
    #         "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    #     }
    #     return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    #
    # # 定义训练数据
    # loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4)
    # if mode == tf.estimator.ModeKeys.TRAIN:
    #
    #     train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    #
    #     return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    #
    # # 定义评估数据
    # accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.nn.softmax(logits), name='accuracy')
    # metrics = {"accuracy": accuracy}
    # tf.summary.scalar('accuracy', accuracy[1])  # 可视化处理
    # if mode == tf.estimator.ModeKeys.EVAL:
    #     return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)
    loss = None
    train_op = None
    if mode != learn.ModeKeys.INFER:
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    if mode == learn.ModeKeys.TRAIN:
        optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    predictions = {
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

    accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.nn.softmax(logits), name='accuracy')
    metrics = {"accuracy": accuracy}

    return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss,
                                   train_op=train_op, eval_metric_ops=metrics)


sess = tf.Session()


def train(train_data, labels):
    # 创建一个占位变量，用于后面的feed_dict中去接受值
    train_init = tf.placeholder(tf.float32, shape=(25000, 64, 64, 3))
    newTrain = tf.Variable(train_init)
    train_data = sess.run(newTrain.assign(train_init), feed_dict={train_init: train_data})
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': train_data}, y=labels, shuffle=True)

    classifier = learn.Estimator(model_fn=catDogModel, model_dir='./checkModels')
    classifier.fit(input_fn=train_input_fn, steps=10000)

    return classifier


def test(test_data, classifier):
    test_init = tf.placeholder(tf.float32, shape=(12500, 64, 64, 3))
    newTest = tf.Variable(test_init)
    test_data = sess.run(newTest.assign(test_init), feed_dict={test_init: test_data})
    test_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': test_data}, shuffle=True)
    predictions = classifier.predict(input_fn=test_input_fn)

    predict = pd.DataFrame({"id": range(len(test_data)), 'label': [i for i in predictions['probabilities']]})

    predict.to_csv('./data/predictions.csv', index=False)


def runCatDog():
    train_data, labels, test_data = loadDataSet()

    classifier = train(train_data, labels)

    test(test_data, classifier)


if __name__ == '__main__':
    runCatDog()