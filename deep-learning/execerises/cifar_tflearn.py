from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
#from read_cifar_data import dict_data

import tensorflow as tf
import numpy as np

#cifar_input_data = dict_data['data'].astype(np.float32)[0:128]

#import tflearn.datasets.oxflower17 as oxflower17
#X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))
#print("InputDataSet Size:", cifar_input_data.shape)
#image_1 = cifar_input_data[0, :]
#print(image_1)
#tempImage = np.zeros((128, 32, 32, 3), dtype=np.float64)
#for i in range(128):
#    imdata = cifar_input_data[i, :]
#    # print(tempImage.shape)
#    tempImage[i, :, :, 0] = imdata[0:1024].reshape((32, 32))
#    tempImage[i, :, :, 1] = imdata[1024:2048].reshape((32, 32))
#    tempImage[i, :, :, 2] = imdata[2048:3072].reshape((32, 32))
    # print(tempImage.dtype)
    # reshaped_image = tf.cast(cifar_input_data.tolist().uint8image, tf.float32)
# print(tempImage[0])
#tf_image = tf.convert_to_tensor(tempImage, dtype=tf.float32)
# print(tf_image)quit
#X= tf.image.resize_bilinear(tempImage, [224, 224])
#Y= tf.image.resize_bilinear(tempImage, [224, 224])
#print(X)




from tflearn.datasets import cifar10
(X, Y), (X_test, Y_test) = cifar10.load_data()
X, Y = shuffle(X, Y)
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Building 'AlexNet'

network = input_data(shape=[None, 224, 224, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
#network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
#network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
#network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 17, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)
model.fit(X, Y, n_epoch=10, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=20, run_id='cifar10_cnn')