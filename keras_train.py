from __future__ import print_function

from keras.datasets import cifar10
from keras.layers import merge, Input
from keras.layers import Add
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

from encoder import encoder
import numpy as np
import os

batch_size = 128
nb_classes = 10
nb_epoch = 200
data_augmentation = False
n = 5  # depth = 6*n + 4
k = 4  # widen factor
save_dir = 'model'
model_name = 'encoding'

# the CIFAR10 images are 32x32 RGB with 10 labels
img_rows, img_cols = 32, 32
img_channels = 3
level = 15


def bottleneck(incoming, count, nb_in_filters, nb_out_filters, dropout=None, subsample=(2, 2)):
    outgoing = wide_basic(incoming, nb_in_filters, nb_out_filters, dropout, subsample)
    for i in range(1, count):
        outgoing = wide_basic(outgoing, nb_out_filters, nb_out_filters, dropout, subsample=(1, 1))

    return outgoing


def wide_basic(incoming, nb_in_filters, nb_out_filters, dropout=None, subsample=(2, 2)):
    nb_bottleneck_filter = nb_out_filters

    if nb_in_filters == nb_out_filters:
        # conv3x3
        y = BatchNormalization(mode=0, axis=1)(incoming)
        y = Activation('relu')(y)
        y = ZeroPadding2D((1, 1))(y)
        y = Convolution2D(nb_bottleneck_filter, nb_row=3, nb_col=3,
                          subsample=subsample, init='he_normal', border_mode='valid')(y)

        # conv3x3
        y = BatchNormalization(mode=0, axis=1)(y)
        y = Activation('relu')(y)
        if dropout is not None:
            y = Dropout(dropout)(y)
        y = ZeroPadding2D((1, 1))(y)
        y = Convolution2D(nb_bottleneck_filter, nb_row=3, nb_col=3,
                          subsample=(1, 1), init='he_normal', border_mode='valid')(y)

        #return merge([incoming, y], mode='sum')
        return Add()([incoming, y])

    else:  # Residual Units for increasing dimensions
        # common BN, ReLU
        shortcut = BatchNormalization(mode=0, axis=1)(incoming)
        shortcut = Activation('relu')(shortcut)

        # conv3x3
        y = ZeroPadding2D((1, 1))(shortcut)
        y = Convolution2D(nb_bottleneck_filter, nb_row=3, nb_col=3,
                          subsample=subsample, init='he_normal', border_mode='valid')(y)

        # conv3x3
        y = BatchNormalization(mode=0, axis=1)(y)
        y = Activation('relu')(y)
        if dropout is not None:
            y = Dropout(dropout)(y)
        y = ZeroPadding2D((1, 1))(y)
        y = Convolution2D(nb_out_filters, nb_row=3, nb_col=3,
                          subsample=(1, 1), init='he_normal', border_mode='valid')(y)

        # shortcut
        shortcut = Convolution2D(nb_out_filters, nb_row=1, nb_col=1,
                                 subsample=subsample, init='he_normal', border_mode='same')(shortcut)

        #return merge([shortcut, y], mode='sum')
        return Add()([shortcut, y])


# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

img_input = Input(shape=(img_rows, img_cols, img_channels * level))

# one conv at the beginning (spatial size: 32x32)
x = ZeroPadding2D((1, 1))(img_input)
x = Convolution2D(16, nb_row=3, nb_col=3)(x)

# Stage 1 (spatial size: 32x32)
x = bottleneck(x, n, 16, 16 * k, dropout=0.3, subsample=(1, 1))
# Stage 2 (spatial size: 16x16)
x = bottleneck(x, n, 16 * k, 32 * k, dropout=0.3, subsample=(2, 2))
# Stage 3 (spatial size: 8x8)
x = bottleneck(x, n, 32 * k, 64 * k, dropout=0.3, subsample=(2, 2))

x = BatchNormalization(mode=0, axis=1)(x)
x = Activation('relu')(x)
x = AveragePooling2D((8, 8), strides=(1, 1))(x)
x = Flatten()(x)
preds = Dense(nb_classes, activation='softmax')(x)

model = Model(input=img_input, output=preds)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

##encoding
X_train = np.transpose(X_train, axes=(0,3,1,2))
encoder = encoder(level=15)
channel0, channel1, channel2 = X_train[:, 0, :, :], X_train[:, 1, :, :], X_train[:, 2, :, :]
channel0, channel1, channel2 = encoder.tempencoding(channel0), encoder.tempencoding(channel1), encoder.tempencoding(
    channel2)

X_train = np.concatenate([channel0, channel1, channel2],axis=1)
X_train =np.transpose(X_train, axes=(0,2,3,1))


X_test = np.transpose(X_test, axes=(0,3,1,2))
channel0, channel1, channel2 = X_test[:, 0, :, :], X_test[:, 1, :, :], X_test[:, 2, :, :]
channel0, channel1, channel2 = encoder.tempencoding(channel0), encoder.tempencoding(channel1), encoder.tempencoding(
    channel2)

X_test = np.concatenate([channel0, channel1, channel2],axis=1)
X_test = np.transpose(X_test, axes=(0,2,3,1))


if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(X_train, Y_train,
                                     batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test))

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])