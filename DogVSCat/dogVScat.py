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
    data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.uint8)
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
    # train_dogs = [TRAIN_DIR + filename for filename in os.listdir(TRAIN_DIR) if 'dog' in filename][:2000]
    # train_cats = [TRAIN_DIR + filename for filename in os.listdir(TRAIN_DIR) if 'cat' in filename][:2000]
    # train_images = train_dogs + train_cats
    test_images = [TEST_DIR + filename for filename in os.listdir(TEST_DIR)]
    random.shuffle(train_images)
    labels = [1 if 'dog' in filename else 0 for filename in train_images]

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
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # 定义训练数据
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4)
    if mode == tf.estimator.ModeKeys.TRAIN:

        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # 定义评估数据
    accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.nn.softmax(logits), name='accuracy')
    metrics = {"accuracy": accuracy}
    tf.summary.scalar('accuracy', accuracy[1])  # 可视化处理
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)


def runCatDog():
    train_data, labels, test_data = loadDataSet()
    train_data = tf.cast(train_data, dtype=tf.float32)
    test_data = tf.cast(test_data, dtype=tf.float32)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': train_data}, y=labels, shuffle=True)
    test_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': test_data}, shuffle=True)

    classifier = learn.Estimator(model_fn=catDogModel, model_dir='./checkModels')

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50
    )

    classifier.fit(input_fn=train_input_fn, steps=10000, monitors=[logging_hook])

    predictions = classifier.predict(input_fn=test_input_fn)

    predict = pd.DataFrame({"id": range(len(test_data)), 'label': predictions['probabilities']})

    predict.to_csv('./data/predictions.csv', index=False)


if __name__ == '__main__':
    runCatDog()