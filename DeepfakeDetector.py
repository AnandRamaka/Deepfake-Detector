import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pickle
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
#import yaml
#import keras

#from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, GlobalMaxPooling1D
from sklearn.model_selection import GridSearchCV
#from keras.wrappers.scikit_learn import KerasClassifier
# import keras.layers as kl
# import keras.initializers as ki
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, GlobalMaxPooling1D
import keras_resnet.models
# import keras

CATEGORIES = ["Fake Faces", "Real Faces"]
IMG_SIZE = 256
batch_size = 256
nb_epoch = 50

X, y = [], []
for i in range(200):
    tempX, tempY = pickle.load( open("dataset" + str(i) + ".p", "rb") )
    #print(tempX)
    X += tempX
    y += tempY
X = np.asarray(X).astype('float32')
y = np.asarray(y).astype('float32')

print("Checkpoint 1")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

# X_train, X_val, X_test = pickle.load(open("Xdata.pickle", "rb"))
# y_train, y_val, y_test = pickle.load(open("Ydata.pickle", "rb"))

from six.moves import cPickle
X_train = X_train.reshape(X_train.shape[0], IMG_SIZE, IMG_SIZE, 1)
X_val = X_val.reshape(X_val.shape[0], IMG_SIZE, IMG_SIZE, 1)
X_test = X_test.reshape(X_test.shape[0], IMG_SIZE, IMG_SIZE, 1)

print("Checkpoint 2")
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')
Y_train = np_utils.to_categorical(y_train, len(CATEGORIES))
Y_val = np_utils.to_categorical(y_val, len(CATEGORIES))
Y_test = np_utils.to_categorical(y_test, len(CATEGORIES))

print("Checkpoint 3")


print(np.shape(Y_test))
use_last_model = False






shape, classes = (IMG_SIZE, IMG_SIZE, 1), 2
x = keras.layers.Input(shape)
model = keras_resnet.models.ResNet2D18(x, classes=classes)
model.compile("adam", "binary_crossentropy", ["accuracy"])

model.load_weights("Best_model.h5")

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
        vertical_flip=False)
datagen.fit(X_train)
#pickle.dump(datagen, open("datagen.pickle", "wb") )

checkpoint_cb = keras.callbacks.ModelCheckpoint("ResNet_model.h5", save_best_only=False, save_freq='epoch')

print("Training started...")

model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        validation_data=(X_val, Y_val),
                        epochs=10, verbose=1,
                        callbacks=[checkpoint_cb])