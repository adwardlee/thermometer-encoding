## setup_cifar.py -- cifar data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import pickle
import urllib.request

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, add
from keras.initializers import he_normal
from keras import regularizers
from keras.utils import np_utils
from keras.models import load_model
from keras.datasets import cifar10

from encoder import encoder


def load_batch(fpath, label_key='labels'):
    f = open(fpath, 'rb')
    d = pickle.load(f, encoding="bytes")
    for k, v in d.items():
        del (d[k])
        d[k.decode("utf8")] = v
    f.close()
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    final = np.zeros((data.shape[0], 32, 32, 3), dtype=np.float32)
    final[:, :, :, 0] = data[:, 0, :, :]
    final[:, :, :, 1] = data[:, 1, :, :]
    final[:, :, :, 2] = data[:, 2, :, :]

    final /= 255
    # final -= .5

    labels2 = np.zeros((len(labels), 10))
    labels2[np.arange(len(labels2)), labels] = 1

    return final, labels


def load_batch(fpath):
    f = open(fpath, "rb").read()
    size = 32 * 32 * 3 + 1
    labels = []
    images = []
    for i in range(10000):
        arr = np.fromstring(f[i * size:(i + 1) * size], dtype=np.uint8)
        lab = np.identity(10)[arr[0]]
        img = arr[1:].reshape((3, 32, 32)).transpose((1, 2, 0))

        labels.append(lab)
        images.append((img / 255) - .5)
    return np.array(images), np.array(labels)


def wide_residual_network(img_input, classes_num, depth, k, weight_decay=0.0005, use_log=True):
    print('Wide-Resnet %dx%d' % (depth, k))
    n_filters = [16, 16 * k, 32 * k, 64 * k]
    n_stack = (depth - 4) / 6
    in_filters = 16

    def conv3x3(x, filters):
        return Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                      kernel_initializer=he_normal(),
                      kernel_regularizer=regularizers.l2(weight_decay))(x)

    def residual_block(x, out_filters, increase_filter=False):
        if increase_filter:
            first_stride = (2, 2)
        else:
            first_stride = (1, 1)
        pre_bn = BatchNormalization()(x)
        pre_relu = Activation('relu')(pre_bn)
        conv_1 = Conv2D(out_filters, kernel_size=(3, 3), strides=first_stride, padding='same',
                        kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(
            pre_relu)
        bn_1 = BatchNormalization()(conv_1)
        relu1 = Activation('relu')(bn_1)
        conv_2 = Conv2D(out_filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(
            relu1)
        if increase_filter or in_filters != out_filters:
            projection = Conv2D(out_filters, kernel_size=(1, 1), strides=first_stride, padding='same',
                                kernel_initializer=he_normal(),
                                kernel_regularizer=regularizers.l2(weight_decay))(x)
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
        return block

    def wide_residual_layer(x, out_filters, increase_filter=False):
        x = residual_block(x, out_filters, increase_filter)
        in_filters = out_filters
        for _ in range(1, int(n_stack)):
            x = residual_block(x, out_filters)
        return x

    x = conv3x3(img_input, n_filters[0])
    x = wide_residual_layer(x, n_filters[1])
    x = wide_residual_layer(x, n_filters[2], increase_filter=True)
    x = wide_residual_layer(x, n_filters[3], increase_filter=True)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    if use_log:
        x = Dense(classes_num, activation='softmax', kernel_initializer=he_normal(),
                  kernel_regularizer=regularizers.l2(weight_decay))(x)
    else:
        x = Dense(classes_num)(x)
    return x


class CIFAR:
    def __init__(self):
        train_data = []
        train_labels = []

        if not os.path.exists("cifar-10-batches-bin"):
            urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
                                       "cifar-data.tar.gz")
            os.popen("tar -xzf cifar-data.tar.gz").read()

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = np.transpose(x_train, axes=(0, 3, 1, 2))
        temp_encoder = encoder(level=15)
        channel0, channel1, channel2 = x_train[:, 0, :, :], x_train[:, 1, :, :], x_train[:, 2, :, :]
        channel0, channel1, channel2 = temp_encoder.tempencoding(channel0), temp_encoder.tempencoding(
            channel1), temp_encoder.tempencoding(
            channel2)

        x_train = np.concatenate([channel0, channel1, channel2], axis=1)
        x_train = np.transpose(x_train, axes=(0, 2, 3, 1))

        x_test = np.transpose(x_test, axes=(0, 3, 1, 2))
        channel0, channel1, channel2 = x_test[:, 0, :, :], x_test[:, 1, :, :], x_test[:, 2, :, :]
        channel0, channel1, channel2 = temp_encoder.tempencoding(channel0), temp_encoder.tempencoding(
            channel1), temp_encoder.tempencoding(
            channel2)

        x_test = np.concatenate([channel0, channel1, channel2], axis=1)
        x_test = np.transpose(x_test, axes=(0, 2, 3, 1))

        self.train_data = x_train
        self.test_data = x_test
        self.train_labels = y_train
        self.test_labels = y_test

        # for i in range(5):
        #     r, s = load_batch("cifar-10-batches-bin/data_batch_" + str(i + 1) + ".bin")
        #     train_data.extend(r)
        #     train_labels.extend(s)
        #
        # train_data = np.array(train_data, dtype=np.float32)
        # train_labels = np.array(train_labels)
        #
        # self.test_data, self.test_labels = load_batch("cifar-10-batches-bin/test_batch.bin")
        #
        # VALIDATION_SIZE = 5000
        #
        # self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        # self.validation_labels = train_labels[:VALIDATION_SIZE]
        # self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        # self.train_labels = train_labels[VALIDATION_SIZE:]
        #
        # ################# encoder train_data
        # self.train_data = np.transpose(self.train_data, axes=(0, 3, 1, 2))
        # temp_encoder = encoder(level=15)
        # channel0, channel1, channel2 = self.train_data[:, 0, :, :], self.train_data[:, 1, :, :], self.train_data[:, 2,
        #                                                                                          :, :]
        # channel0, channel1, channel2 = temp_encoder.tempencoding(channel0), temp_encoder.tempencoding(
        #     channel1), temp_encoder.tempencoding(
        #     channel2)
        # self.train_data = np.concatenate([channel0, channel1, channel2], axis=1)
        # self.train_data = np.transpose(self.train_data, axes=(0, 2, 3, 1))
        #
        # ################# encoder validation_data
        # self.validation_data = np.transpose(self.validation_data, axes=(0, 3, 1, 2))
        # temp_encoder = encoder(level=15)
        # channel0, channel1, channel2 = self.validation_data[:, 0, :, :], self.validation_data[:, 1, :,
        #                                                                  :], self.validation_data[:, 2,
        #                                                                      :, :]
        # channel0, channel1, channel2 = temp_encoder.tempencoding(channel0), temp_encoder.tempencoding(
        #     channel1), temp_encoder.tempencoding(
        #     channel2)
        # self.validation_data = np.concatenate([channel0, channel1, channel2], axis=1)
        # self.validation_data = np.transpose(self.validation_data, axes=(0, 2, 3, 1))
        #
        # ################# encoder test_data
        # self.test_data = np.transpose(self.test_data, axes=(0, 3, 1, 2))
        # temp_encoder = encoder(level=15)
        # channel0, channel1, channel2 = self.test_data[:, 0, :, :], self.test_data[:, 1, :, :], self.test_data[:, 2,
        #                                                                                        :, :]
        # channel0, channel1, channel2 = temp_encoder.tempencoding(channel0), temp_encoder.tempencoding(
        #     channel1), temp_encoder.tempencoding(
        #     channel2)
        # self.test_data = np.concatenate([channel0, channel1, channel2], axis=1)
        # self.test_data = np.transpose(self.test_data, axes=(0, 2, 3, 1))


class CIFARModel:
    def __init__(self, restore=None, session=None, use_log=False):
        self.num_channels = 3
        self.image_size = 32
        self.num_labels = 10

        model = Sequential()

        model.add(Conv2D(64, (3, 3),
                         input_shape=(32, 32, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(10))
        if use_log:
            model.add(Activation('softmax'))
        if restore:
            model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model(data)


class CIFAR_WIDE:
    def __init__(self, restore=None, session=None, use_log=True, k=4, depth=34):
        self.num_channels = 45
        self.image_size = 32
        self.num_labels = 10

        img_input = Input(shape=(self.image_size, self.image_size, self.num_channels))

        output = wide_residual_network(img_input, self.num_labels, depth, k, use_log)
        model = Model(img_input, output)

        if restore:
            model.load_weights(restore)
        self.model = model

    def predict(self, data):
        return self.model(data)

BATCH_SIZE = 1

with tf.Session() as sess:
    #data, model = MNIST(), MNISTModel("models/mnist", sess)
    data, model = CIFAR(), CIFAR_WIDE("models/wide_resnet.h5", sess)
    #data, model = ImageNet(), InceptionModel(sess)

    x = tf.placeholder(tf.float32, (None, model.image_size, model.image_size, model.num_channels))
    y = model.predict(x)

    r = []
    for i in range(0,len(data.test_data),BATCH_SIZE):
        pred = sess.run(y, {x: data.test_data[i:i+BATCH_SIZE]})
        #print(pred)
        #print('real',data.test_labels[i],'pred',np.argmax(pred))
        r.append(np.argmax(pred,1) == np.argmax(data.test_labels[i:i+BATCH_SIZE],1))
        print(np.mean(r))