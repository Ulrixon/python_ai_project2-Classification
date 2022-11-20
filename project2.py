#%%
import imp
from random import random
from re import S
import numpy as np
import matplotlib.pyplot as plt
from pandas import array
import sklearn
import tensorflow as tf
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense
from keras.utils.vis_utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
from PIL import Image
from keras.utils import np_utils
#%%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)
np.unique(y_train)
x_train=x_train/255
x_test=x_test/255

y_train_matrix=np_utils.to_categorical(y_train)
y_test_matrix=np_utils.to_categorical(y_test)
# %%
for i in range(3):
    im=Image.fromarray(x_train[i])
    plt.imshow(im)
    plt.show()
# %%
#%%






initializer1 = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
initializer2 = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
initializer3 = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
model_input = Input(shape=(32,32,3))
#rescale=tf.keras.layers.Rescaling(scale=1./255)(model_input)
#shift=tf.keras.layers.RandomTranslation(0.1,0.1,fill_mode="constant")(model_input)
#rotate=tf.keras.layers.RandomRotation(factor=(0.1),fill_mode="constant")(shift)
#zoom=tf.keras.layers.RandomZoom(0.1,fill_mode="constant")(rotate)
#shear=tf.keras.preprocessing.image.random_shear(0.1,fill_mode='nearest')(zoom)
#a = tf.keras.layers.BatchNormalization()(model_input, training=True)
conv1=tf.keras.layers.Conv2D(64,(5,5), activation='relu', padding="same")(model_input)
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(conv1)
#b = tf.keras.layers.BatchNormalization()(pool1, training=True)
conv2=tf.keras.layers.Conv2D(128,(5,5), activation='relu', padding="same")(pool1)
pool2=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(conv2)
flatten1= tf.keras.layers.Flatten()(pool2)

conv3=tf.keras.layers.Conv2D(64,(3,3), activation='relu', padding="same")(model_input)
pool3=tf.keras.layers.MaxPooling2D(
    pool_size=(4, 4), strides=None, padding="valid", data_format=None,
  )(conv3)

flatten2= tf.keras.layers.Flatten()(pool3)



#conv4=tf.keras.layers.Conv2D(32,(7,7), activation='relu', padding="same")(model_input)

#pool4=tf.keras.layers.MaxPooling2D(
#    pool_size=(4, 4), strides=None, padding="valid", data_format=None,
#  )(conv4)
#flatten3=tf.keras.layers.Flatten()(pool4)

concate=tf.keras.layers.Concatenate(axis=1)([flatten1,flatten2])

dense1=tf.keras.layers.Dense(800, activation='relu',kernel_initializer='random_normal',
            bias_initializer=initializer1)(concate)
dropout1=tf.keras.layers.Dropout(0.5)(dense1)
dense2=tf.keras.layers.Dense(400, activation='relu',kernel_initializer='random_normal', 
                bias_initializer=initializer2)(dropout1)
dropout2=tf.keras.layers.Dropout(0.5)(dense2)
dense3=tf.keras.layers.Dense(100, activation='softmax',kernel_initializer='random_normal',
 bias_initializer=initializer3)(dropout2)


model = Model(inputs=model_input, outputs=dense3)

loss_fn=tf.keras.losses.CategoricalCrossentropy()

adam =tf.keras.optimizers.Adam(
    learning_rate=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam",
    
)

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer=adam,
              loss=loss_fn,
              metrics=['accuracy'])

# %%
model.fit(x_train,y_train_matrix,
  validation_data = (x_test,y_test_matrix),epochs=100,verbose=1,batch_size=1000,shuffle=True, use_multiprocessing=True,workers=8)

#%%
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()