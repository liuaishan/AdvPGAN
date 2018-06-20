"""VGG16 model for Keras.
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
"""

import warnings
from keras.applications import vgg16
from keras.models import Model
from keras import optimizers
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping

import os
import h5py
import cv2
import numpy as np
from timeit import default_timer as timer

import pickle
from sklearn.preprocessing import OneHotEncoder

start = timer()

def OHE_labels(Y_tr, N_classes):
    OHC = OneHotEncoder()
    Y_ohc = OHC.fit(np.arange(N_classes).reshape(-1, 1))
    Y_labels = Y_ohc.transform(Y_tr.reshape(-1, 1)).toarray()
    return Y_labels


def pre_process_image(image):
    return image/255.


def load_image(file_path, N_classes, one_hot=True):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    image = data['data']
    if(one_hot):
        label = OHE_labels(data['labels'], N_classes)
    else:
        label = data['labels']

    image = np.array([pre_process_image(image[i]) for i in range(np.shape(image)[0])], dtype=np.float32)
    label = label.astype(np.float32)

    return image, label
    

def VGG16_Model(img_rows=224, img_cols=224, train=False):
    if K.image_data_format() == 'channels_first':
        shape_ord = (3, img_rows, img_cols)
    else:  # channel_last
        shape_ord = (img_rows, img_cols, 3)

    vgg16_model = vgg16.VGG16(weights=None, include_top=False, input_tensor=Input(shape_ord))
    # vgg16_model.summary()

    for layer in vgg16_model.layers:
        layer.trainable = train  # freeze layer

    #add last fully-connected layers
    x = Flatten(input_shape=vgg16_model.output.shape)(vgg16_model.output)
    x = Dense(4096, activation='relu', name='ft_fc1')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    predictions = Dense(43, activation='softmax')(x)

    model = Model(inputs=vgg16_model.input, outputs=predictions)

    #compile the model
    model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                loss='categorical_crossentropy', metrics=['accuracy'])

    for layer in model.layers:
        layer.trainable = train  # freeze layer

    return model


def VGG16_train(train_data_dir, test_data_dir, weigths_dir, N_classes, epochs, batch_size):
    # build model
    model = VGG16_Model(img_rows=128, img_cols=128, train=True)
    # load training data
    train, train_labels = load_image(train_data_dir, N_classes, one_hot=True)
    print("Training images loaded.")
    # fit the model
    start_training = timer()
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model.fit(train, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[early_stopping])
    end_training = timer()
    print("Model fitted.")
    # save trained weights
    model.save_weights(weigths_dir, overwrite=True)

    # load test data
    test, test_labels = load_image(test_data_dir, N_classes, one_hot=False)
    print("Test images loaded.")
    # predcit labels of test images with new weights
    probs = model.predict(test)
    print("Labels predicted.")
    # calculate accuracy
    predicted_labels = []
    for i in range(np.shape(test_labels)[0]):
        cls_prob = probs[i]
        predicted_labels.append(np.argmax(cls_prob))
    acc = np.mean(np.cast['float32'](np.equal(test_labels, predicted_labels)))
    print("accuracy on test: " + str(acc))

    end = timer()
    print("Training time: ", end_training - start_training)
    print("Total time: ", end - start)


def VGG16_predict(test_data_dir, weigths_dir, N_classes):
    K.set_learning_phase(False)
    # build model
    model = VGG16_Model(img_rows=128, img_cols=128, train=False)
    # load weights
    model.load_weights(weigths_dir)
    # load test data
    test, test_labels = load_image(test_data_dir, N_classes, one_hot=False)
    print("Test images loaded.")
    # predcit labels of test images with new weights
    probs = model.predict(test)
    print("Labels predicted.")
    # calculate accuracy
    predicted_labels = []
    for i in range(np.shape(test_labels)[0]):
        cls_prob = probs[i]
        predicted_labels.append(np.argmax(cls_prob))
    acc = np.mean(np.cast['float32'](np.equal(test_labels, predicted_labels)))
    print("accuracy on test: " + str(acc))

    end = timer()
    print("Total time: ", end - start)



if __name__ == "__main__":
    train_data_dir = '/media/dsgDisk/dsgPrivate/liuaishan/GTSRB/data/train.p'
    test_data_dir = '/media/dsgDisk/dsgPrivate/liuaishan/GTSRB/data/test.p'
    valid_data_dir = '/home/zhenxt/zal/GTSRB/data/validation_img_1_8.p'
    weights_dir = '/home/zhenxt/zal/GTSRB/vgg16/'
    N_classes = 43
    batch_size = 16
    epochs = 10000
    weights_dir = weights_dir + 'vgg16_training' + str(epochs) + '.h5'

    # VGG16_train(train_data_dir, test_data_dir, weights_dir, N_classes, epochs, batch_size)
    VGG16_predict(valid_data_dir, weights_dir, 43)

