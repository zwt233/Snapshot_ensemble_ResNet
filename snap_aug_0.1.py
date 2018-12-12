import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Input
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.callbacks import ModelCheckpoint
import keras
from snapshot import SnapshotCallbackBuilder
from keras.models import Model
from keras.regularizers import l2
from keras.datasets import cifar100
from keras import backend as K
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)
from keras.models import load_model
KTF.set_session(sess)


def load_data():
    with tf.name_scope('input'):
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
    return (x_train, y_train), (x_test, y_test)

def preprocess(x_train, y_train, x_test, y_test, substract_pixel_mean=False):
    with tf.name_scope('Preprocess'):
        num_classes = 100
        # Convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        # preprocess data
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        if substract_pixel_mean:
            x_train_mean = np.mean(x_train, axis=0)
            x_train -= x_train_mean
            x_test -= x_train_mean
    return (x_train, y_train), (x_test, y_test)

def build_resnet(x_train,  n):
    depth = n * 6 + 2
    input_shape = x_train.shape[1:]

    def resnet_block(inputs,
                     num_filters=16,
                     kernel_size=3,
                     strides=1,
                     activation='relu'):
        """2D Convolution-Batch Normalization-Activation stack builder

        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            num_filters (int): Conv2D number of filters
            kernel_size (int): Conv2D square kernel dimensions
            strides (int): Conv2D square stride dimensions
            activation (string): activation name

        # Returns
            x (tensor): tensor as input to the next layer
        """
        x = Conv2D(num_filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4))(inputs)
        x = BatchNormalization()(x)
        if activation:
            x = Activation(activation)(x)
        return x


    def resnet_v1(input_shape, depth, num_classes=100):
        """ResNet Version 1 Model builder [a]

        Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
        Last ReLU is after the shortcut connection.
        The number of filters doubles when the feature maps size
        is halved.
        The Number of parameters is approx the same as Table 6 of [a]:
        ResNet20 0.27M
        ResNet32 0.46M
        ResNet44 0.66M
        ResNet56 0.85M
        ResNet110 1.7M

        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)

        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        # Start model definition.
        inputs = Input(shape=input_shape)
        num_filters = 16
        num_sub_blocks = int((depth - 2) / 6)

        x = resnet_block(inputs=inputs)
        # Instantiate convolutional base (stack of blocks).
        for i in range(3):
            for j in range(num_sub_blocks):
                strides = 1
                is_first_layer_but_not_first_block = j == 0 and i > 0
                if is_first_layer_but_not_first_block:
                    strides = 2
                y = resnet_block(inputs=x,
                                 num_filters=num_filters,
                                 strides=strides)
                y = resnet_block(inputs=y,
                                 num_filters=num_filters,
                                 activation=None)
                if is_first_layer_but_not_first_block:
                    x = resnet_block(inputs=x,
                                     num_filters=num_filters,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None)
                x = keras.layers.add([x, y])
                x = Activation('relu')(x)
            num_filters = 2 * num_filters

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)
        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        return model

    model = resnet_v1(input_shape=input_shape, depth=depth)

    # initiate RMSprop optimizer
    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print('start')
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    name_prefix = "cnn-snapshot"
    M = 10
    epochs = 400
    batch_size = 64
    alpha_zero = 0.1
    num_classes = 100
    K = float(num_classes)
    (x_train, y_train), (x_test, y_test) = load_data()  # cifar-100
    (x_train, y_train), (x_test, y_test) = preprocess(x_train, y_train, x_test, y_test)
    # 1、tensorboard显示
    tbCallBack = keras.callbacks.TensorBoard(log_dir='Snapshot ensemble')

    model = build_resnet(x_train, n=5)
    T = epochs
    snapshot = SnapshotCallbackBuilder(T, M, alpha_zero)

    # 数据增强
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False)
    datagen.fit(x_train)
    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).

    callbacks_list = snapshot.get_callbacks(model_prefix=name_prefix)
    # 建模
    with tf.name_scope('Fit'):
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=int(np.ceil(x_train.shape[0] / float(batch_size))),
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            callbacks=callbacks_list)

    # 检验最后一次
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

