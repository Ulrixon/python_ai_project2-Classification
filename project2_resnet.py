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
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
from PIL import Image
from keras.utils import np_utils
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
#from gradient_accumulator.GAModelWrapper import GAModelWrapper
from tensorflow.keras import mixed_precision
tf.keras.mixed_precision.set_global_policy("mixed_float16")
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

#%%
# 在开启对话session前，先创建一个 tf.ConfigProto() 实例对象

gpuConfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
gpuConfig.gpu_options.allow_growth=True
# 限制一个进程使用 60% 的显存


# 把你的配置部署到session  变量名 sess 无所谓
sess1 =tf.compat.v1.Session(config=gpuConfig)


# %% fit
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#early stopping to monitor the validation loss and avoid overfitting
early_stop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=30, restore_best_weights=True)

#reducing learning rate on plateau
rlrop = ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience= 10, factor= 0.5, min_lr= 1e-8, verbose=1)
#X_train, X_validation, y_train, y_validation = train_test_split(x_train, y_train_matrix, test_size=0.2, random_state=93)

train_datagen = ImageDataGenerator(
    #rotation_range=20,
    #horizontal_flip=True,
    #vertical_flip=True
)

model.fit(x=train_datagen.flow(x_train, y_train_matrix,batch_size=25),
  validation_data = (x_test,y_test_matrix),epochs=300,verbose=1,batch_size=50,

  shuffle=True, use_multiprocessing=False,workers=6,callbacks = [early_stop, rlrop])


#%% plot learning curve
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#%% save model
model.save("/mnt/c/Users/ryan7/Documents/GitHub/python_ai_project2/my_model")

#%% resnet_stage4_improve_imagegenerator_parameterx2

model_input = Input(shape=(32,32,3))
rotation=tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)(model_input)
flip=tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(rotation)

conv1=tf.keras.layers.Conv2D(256,(3,3), padding="same")(flip)
batch_norm=tf.keras.layers.BatchNormalization()(conv1)
relu=tf.keras.layers.ReLU()(batch_norm)

#stage1
conv2=tf.keras.layers.Conv2D(256,(3,3), padding="same")(relu)
conv2=tf.keras.layers.BatchNormalization()(conv2)
conv2=tf.keras.layers.ReLU()(conv2)
#dense1=tf.keras.layers.Dropout(0.2)(conv2)

conv2=tf.keras.layers.Conv2D(256,(3,3), padding="same")(conv2)
conv2=tf.keras.layers.BatchNormalization()(conv2)
conv2=tf.keras.layers.ReLU()(conv2)
#dense1=tf.keras.layers.Dropout(0.2)(conv2)

concate=tf.keras.layers.add([relu,conv2])
batch_norm2=tf.keras.layers.BatchNormalization()(concate)

pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(batch_norm2)
dense1=tf.keras.layers.Dropout(0.2)(conv2)
conv3=tf.keras.layers.Conv2D(256,(1,1), padding="same")(dense1)
batch_norm3=tf.keras.layers.BatchNormalization()(conv3)


#stage2
conv2=tf.keras.layers.Conv2D(256,(3,3), padding="same")(batch_norm3)
batch_norm2=tf.keras.layers.BatchNormalization()(conv2)
relu2=tf.keras.layers.ReLU()(batch_norm2)
#dense1=tf.keras.layers.Dropout(0.2)(relu2)

conv2=tf.keras.layers.Conv2D(256,(3,3), padding="same")(relu2)
batch_norm2=tf.keras.layers.BatchNormalization()(conv2)
relu2=tf.keras.layers.ReLU()(batch_norm2)
#dense1=tf.keras.layers.Dropout(0.2)(relu2)

concate=tf.keras.layers.Add()([batch_norm3,relu2])
batch_norm2=tf.keras.layers.BatchNormalization()(concate)

pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(batch_norm2)
#bottle=tf.keras.layers.Conv2D(128,(1,1), padding="same")(pool1)
dense1=tf.keras.layers.Dropout(0.2)(pool1)

conv3=tf.keras.layers.Conv2D(512,(1,1), padding="same")(dense1)
batch_norm3=tf.keras.layers.BatchNormalization()(conv3)

#stage3

conv2=tf.keras.layers.Conv2D(512,(3,3), padding="same")(batch_norm3)
batch_norm2=tf.keras.layers.BatchNormalization()(conv2)
relu2=tf.keras.layers.ReLU()(batch_norm2)
#dense1=tf.keras.layers.Dropout(0.2)(relu2)

conv2=tf.keras.layers.Conv2D(512,(3,3), padding="same")(relu2)
batch_norm2=tf.keras.layers.BatchNormalization()(conv2)
relu2=tf.keras.layers.ReLU()(batch_norm2)
#dense1=tf.keras.layers.Dropout(0.2)(relu2)

concate=tf.keras.layers.Add()([relu2,batch_norm3])
batch_norm2=tf.keras.layers.BatchNormalization()(concate)

pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(batch_norm2)
dense1=tf.keras.layers.Dropout(0.2)(pool1)

conv3=tf.keras.layers.Conv2D(512,(1,1), padding="same")(dense1)
batch_norm3=tf.keras.layers.BatchNormalization()(conv3)

#stage4
conv2=tf.keras.layers.Conv2D(512,(3,3), padding="same")(batch_norm3)
batch_norm2=tf.keras.layers.BatchNormalization()(conv2)
relu2=tf.keras.layers.ReLU()(batch_norm2)
#dense1=tf.keras.layers.Dropout(0.2)(relu2)

conv2=tf.keras.layers.Conv2D(512,(3,3), padding="same")(relu2)
batch_norm2=tf.keras.layers.BatchNormalization()(conv2)
relu2=tf.keras.layers.ReLU()(batch_norm2)
#dense1=tf.keras.layers.Dropout(0.2)(relu2)

concate=tf.keras.layers.Add()([batch_norm3,relu2])
batch_norm2=tf.keras.layers.BatchNormalization()(concate)

pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(batch_norm2)
dense1=tf.keras.layers.Dropout(0.2)(pool1)
flatten1= tf.keras.layers.Flatten()(dense1)

dense1=tf.keras.layers.Dense(1024, activation='relu',kernel_initializer='random_normal'
            )(flatten1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense1=tf.keras.layers.Dense(512, activation='relu',kernel_initializer='random_normal'
            )(dense1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense3=tf.keras.layers.Dense(100, activation='softmax',kernel_initializer='random_normal'
 )(dense1)



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
print(model.summary())


# %%
#%%  resnet_stage4_improve_imagegenerator_parameterx2_relu_change

model_input = Input(shape=(32,32,3))
rotation=tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)(model_input)
flip=tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(rotation)

conv1=tf.keras.layers.Conv2D(256,(3,3), padding="same")(flip)
batch_norm=tf.keras.layers.BatchNormalization()(conv1)
relu=tf.keras.layers.ReLU()(batch_norm)

#stage1
conv2=tf.keras.layers.Conv2D(256,(3,3), padding="same")(relu)
conv2=tf.keras.layers.BatchNormalization()(conv2)
conv2=tf.keras.layers.ReLU()(conv2)
#dense1=tf.keras.layers.Dropout(0.2)(conv2)

conv2=tf.keras.layers.Conv2D(256,(3,3), padding="same")(conv2)
conv2=tf.keras.layers.BatchNormalization()(conv2)
conv2=tf.keras.layers.ReLU()(conv2)
#dense1=tf.keras.layers.Dropout(0.2)(conv2)

conv2=tf.keras.layers.Conv2D(256,(3,3), padding="same")(conv2)
conv2=tf.keras.layers.BatchNormalization()(conv2)

concate=tf.keras.layers.add([relu,conv2])
conv2=tf.keras.layers.ReLU()(concate)

pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(conv2)
#dense1=tf.keras.layers.Dropout(0.2)(conv2)


#stage2
conv3=tf.keras.layers.Conv2D(256,(1,1), padding="same")(pool1)
batch_norm3=tf.keras.layers.BatchNormalization()(conv3)
relu2=tf.keras.layers.ReLU()(batch_norm3)


conv2=tf.keras.layers.Conv2D(256,(3,3), padding="same")(relu2)
batch_norm2=tf.keras.layers.BatchNormalization()(conv2)
relu2=tf.keras.layers.ReLU()(batch_norm2)
#dense1=tf.keras.layers.Dropout(0.2)(relu2)

conv2=tf.keras.layers.Conv2D(256,(3,3), padding="same")(relu2)
batch_norm2=tf.keras.layers.BatchNormalization()(conv2)

#dense1=tf.keras.layers.Dropout(0.2)(relu2)

concate=tf.keras.layers.Add()([pool1,batch_norm2])

relu2=tf.keras.layers.ReLU()(concate)
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(relu2)
#bottle=tf.keras.layers.Conv2D(128,(1,1), padding="same")(pool1)
#dense1=tf.keras.layers.Dropout(0.2)(pool1)



#stage3
conv3=tf.keras.layers.Conv2D(512,(1,1), padding="same")(pool1)
batch_norm3=tf.keras.layers.BatchNormalization()(conv3)
relu2=tf.keras.layers.ReLU()(batch_norm3)


conv2=tf.keras.layers.Conv2D(512,(3,3), padding="same")(relu2)
batch_norm2=tf.keras.layers.BatchNormalization()(conv2)
relu2=tf.keras.layers.ReLU()(batch_norm2)
#dense1=tf.keras.layers.Dropout(0.2)(relu2)

conv2=tf.keras.layers.Conv2D(512,(3,3), padding="same")(relu2)
batch_norm2=tf.keras.layers.BatchNormalization()(conv2)

#dense1=tf.keras.layers.Dropout(0.2)(relu2)
conv2=tf.keras.layers.Conv2D(512,(3,3), padding="same")(pool1)
batch_norm3=tf.keras.layers.BatchNormalization()(conv2)

concate=tf.keras.layers.Add()([batch_norm2,batch_norm3])

relu2=tf.keras.layers.ReLU()(concate)
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(relu2)

#dense1=tf.keras.layers.Dropout(0.2)(pool1)

#stage4

conv3=tf.keras.layers.Conv2D(512,(1,1), padding="same")(pool1)
batch_norm3=tf.keras.layers.BatchNormalization()(conv3)
relu2=tf.keras.layers.ReLU()(batch_norm3)

conv2=tf.keras.layers.Conv2D(512,(3,3), padding="same")(relu2)
batch_norm2=tf.keras.layers.BatchNormalization()(conv2)
relu2=tf.keras.layers.ReLU()(batch_norm2)
#dense1=tf.keras.layers.Dropout(0.2)(relu2)

conv2=tf.keras.layers.Conv2D(512,(3,3), padding="same")(relu2)
batch_norm2=tf.keras.layers.BatchNormalization()(conv2)

#dense1=tf.keras.layers.Dropout(0.2)(relu2)

concate=tf.keras.layers.Add()([pool1,batch_norm2])
relu2=tf.keras.layers.ReLU()(concate)

pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(relu2)
#dense1=tf.keras.layers.Dropout(0.2)(pool1)
flatten1= tf.keras.layers.Flatten()(pool1)

dense1=tf.keras.layers.Dense(1024, activation='relu',kernel_initializer='random_normal'
            )(flatten1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense1=tf.keras.layers.Dense(512, activation='relu',kernel_initializer='random_normal'
            )(dense1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense3=tf.keras.layers.Dense(100, activation='softmax',kernel_initializer='random_normal'
 )(dense1)



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
print(model.summary())



# %% resnet_stage4_block

model_input = Input(shape=(32,32,3))
rotation=tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)(model_input)
flip=tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(rotation)

conv1=tf.keras.layers.Conv2D(256,(3,3), padding="same")(flip)
batch_norm=tf.keras.layers.BatchNormalization()(conv1)
relu=tf.keras.layers.ReLU()(batch_norm)

#stage1
conv2=tf.keras.layers.Conv2D(256,(3,3), padding="same")(relu)
conv2=tf.keras.layers.BatchNormalization()(conv2)
conv2=tf.keras.layers.ReLU()(conv2)
#dense1=tf.keras.layers.Dropout(0.2)(conv2)

conv2=tf.keras.layers.Conv2D(256,(3,3), padding="same")(conv2)
conv2=tf.keras.layers.BatchNormalization()(conv2)
conv2=tf.keras.layers.ReLU()(conv2)
#dense1=tf.keras.layers.Dropout(0.2)(conv2)

conv2=tf.keras.layers.Conv2D(256,(3,3), padding="same")(conv2)
conv2=tf.keras.layers.BatchNormalization()(conv2)

concate=tf.keras.layers.add([relu,conv2])
conv2=tf.keras.layers.ReLU()(concate)

pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(conv2)
#dense1=tf.keras.layers.Dropout(0.2)(conv2)


#stage2
conv3=tf.keras.layers.Conv2D(256,(1,1), padding="same")(pool1)
batch_norm3=tf.keras.layers.BatchNormalization()(conv3)
relu2=tf.keras.layers.ReLU()(batch_norm3)


conv2=tf.keras.layers.Conv2D(256,(3,3), padding="same")(relu2)
batch_norm2=tf.keras.layers.BatchNormalization()(conv2)
relu2=tf.keras.layers.ReLU()(batch_norm2)
#dense1=tf.keras.layers.Dropout(0.2)(relu2)

conv2=tf.keras.layers.Conv2D(256,(3,3), padding="same")(relu2)
batch_norm2=tf.keras.layers.BatchNormalization()(conv2)

#dense1=tf.keras.layers.Dropout(0.2)(relu2)

concate=tf.keras.layers.Add()([pool1,batch_norm2])

relu2=tf.keras.layers.ReLU()(concate)
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(relu2)
#bottle=tf.keras.layers.Conv2D(128,(1,1), padding="same")(pool1)
#dense1=tf.keras.layers.Dropout(0.2)(pool1)



#stage3
conv3=tf.keras.layers.Conv2D(512,(1,1), padding="same")(pool1)
batch_norm3=tf.keras.layers.BatchNormalization()(conv3)
relu2=tf.keras.layers.ReLU()(batch_norm3)


conv2=tf.keras.layers.Conv2D(512,(3,3), padding="same")(relu2)
batch_norm2=tf.keras.layers.BatchNormalization()(conv2)
relu2=tf.keras.layers.ReLU()(batch_norm2)
#dense1=tf.keras.layers.Dropout(0.2)(relu2)

conv2=tf.keras.layers.Conv2D(512,(3,3), padding="same")(relu2)
batch_norm2=tf.keras.layers.BatchNormalization()(conv2)

#dense1=tf.keras.layers.Dropout(0.2)(relu2)
conv2=tf.keras.layers.Conv2D(512,(3,3), padding="same")(pool1)
batch_norm3=tf.keras.layers.BatchNormalization()(conv2)

concate=tf.keras.layers.Add()([batch_norm2,batch_norm3])

relu2=tf.keras.layers.ReLU()(concate)
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(relu2)

#dense1=tf.keras.layers.Dropout(0.2)(pool1)

#stage4

conv3=tf.keras.layers.Conv2D(1024,(3,3), padding="same")(pool1)
batch_norm3=tf.keras.layers.BatchNormalization()(conv3)
relu2=tf.keras.layers.ReLU()(batch_norm3)

conv2=tf.keras.layers.Conv2D(1024,(3,3), padding="same")(relu2)
batch_norm2=tf.keras.layers.BatchNormalization()(conv2)
relu2=tf.keras.layers.ReLU()(batch_norm2)
#dense1=tf.keras.layers.Dropout(0.2)(relu2)

conv2=tf.keras.layers.Conv2D(1024,(3,3), padding="same")(relu2)
batch_norm2=tf.keras.layers.BatchNormalization()(conv2)

#dense1=tf.keras.layers.Dropout(0.2)(relu2)
conv2=tf.keras.layers.Conv2D(1024,(3,3), padding="same")(pool1)
batch_norm3=tf.keras.layers.BatchNormalization()(conv2)

concate=tf.keras.layers.Add()([batch_norm3,batch_norm2])
relu2=tf.keras.layers.ReLU()(concate)

pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(relu2)
#dense1=tf.keras.layers.Dropout(0.2)(pool1)




flatten1= tf.keras.layers.Flatten()(pool1)

dense1=tf.keras.layers.Dense(1024, activation='relu',kernel_initializer='random_normal'
            )(flatten1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense1=tf.keras.layers.Dense(512, activation='relu',kernel_initializer='random_normal'
            )(dense1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense3=tf.keras.layers.Dense(100, activation='softmax',kernel_initializer='random_normal'
 )(dense1)



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
print(model.summary())




# %% resnet_stage4_12layer


def efficeint_block_same_channel(x,expand,squeeze,block_name):
    with tf.name_scope(block_name):
        m=tf.keras.layers.Conv2D(expand,(1,1), padding= "same")(x)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3),strides=(1, 1), padding="same")(m)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.Conv2D(squeeze,(1,1), padding= "same")(m)
        m=tf.keras.layers.BatchNormalization()(m)

        return tf.keras.layers.Add()([m,x])


def efficeint_block_different_channel(x,expand,squeeze,block_name):
    with tf.name_scope(block_name):
        m=tf.keras.layers.Conv2D(expand,(1,1),padding= "same")(x)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3),strides=(1, 1),padding="same")(m)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.Conv2D(squeeze,(1,1),padding= "same")(m)
        m=tf.keras.layers.BatchNormalization()(m)

        z=tf.keras.layers.Conv2D(squeeze,(1,1),padding= "same")(x)
        z=tf.keras.layers.BatchNormalization()(z)

        return tf.keras.layers.Add()([m,z])

model_input = Input(shape=(32,32,3))
#rotation=tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)(model_input)
#flip=tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(rotation)

conv1=tf.keras.layers.Conv2D(128,(3,3), padding="same")(model_input)
batch_norm=tf.keras.layers.BatchNormalization()(conv1)
relu=tf.keras.layers.ReLU()(batch_norm)

stage1_block1=efficeint_block_same_channel(relu,128,128,"stage1_block1")
stage1_block2=efficeint_block_same_channel(stage1_block1,768,128,"stage1_block2")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage1_block2)


stage2_block1=efficeint_block_different_channel(pool1,1536,256,"stage2_block1")
stage2_block2=efficeint_block_same_channel(stage2_block1,1536,256,"stage2_block2")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage2_block2)


stage3_block1=efficeint_block_different_channel(pool1,3072,512,"stage3_block1")
stage3_block2=efficeint_block_same_channel(stage3_block1,3072,512,"stage3_block2")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage3_block2)


stage4_block1=efficeint_block_different_channel(pool1,6144,1024,"stage4_block1")
stage4_block2=efficeint_block_same_channel(stage4_block1,6144,1024,"stage4_block2")

pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage4_block2)

flatten1= tf.keras.layers.Flatten()(pool1)

dense1=tf.keras.layers.Dense(1024, activation='relu',kernel_initializer='random_normal'
            )(flatten1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense1=tf.keras.layers.Dense(512, activation='relu',kernel_initializer='random_normal'
            )(dense1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense3=tf.keras.layers.Dense(100, activation='softmax',kernel_initializer='random_normal'
 )(dense1)

 
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

#opt = AdamAccumulate(lr=0.0001, decay=1e-5, accum_iters=5)


plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer=adam,
              loss=loss_fn,
              metrics=['accuracy'])
print(model.summary())

# %% adam accumulation

from keras.optimizers import Optimizer
import keras.backend as K
class AccumOptimizer(Optimizer):
    """Inheriting Optimizer class, wrapping the original optimizer
    to achieve a new corresponding optimizer of gradient accumulation.
    # Arguments
        optimizer: an instance of keras optimizer (supporting
                    all keras optimizers currently available);
        steps_per_update: the steps of gradient accumulation
    # Returns
        a new keras optimizer.
    """
    def __init__(self, optimizer, steps_per_update=1, **kwargs):
        super(AccumOptimizer, self).__init__(**kwargs)
        self.optimizer = optimizer
        with K.name_scope(self.__class__.__name__):
            self.steps_per_update = steps_per_update
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.cond = K.equal(self.iterations % self.steps_per_update, 0)
            self.lr = self.optimizer.lr
            self.optimizer.lr = K.switch(self.cond, self.optimizer.lr, 0.)
            for attr in ['momentum', 'rho', 'beta_1', 'beta_2']:
                if hasattr(self.optimizer, attr):
                    value = getattr(self.optimizer, attr)
                    setattr(self, attr, value)
                    setattr(self.optimizer, attr, K.switch(self.cond, value, 1 - 1e-7))
            for attr in self.optimizer.get_config():
                if not hasattr(self, attr):
                    value = getattr(self.optimizer, attr)
                    setattr(self, attr, value)
            # Cover the original get_gradients method with accumulative gradients.
            def get_gradients(loss, params):
                return [ag / self.steps_per_update for ag in self.accum_grads]
            self.optimizer.get_gradients = get_gradients
    def get_updates(self, loss, params):
        self.updates = [
            K.update_add(self.iterations, 1),
            K.update_add(self.optimizer.iterations, K.cast(self.cond, 'int64')),
        ]
        # gradient accumulation
        self.accum_grads = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        grads = self.get_gradients(loss, params)
        for g, ag in zip(grads, self.accum_grads):
            self.updates.append(K.update(ag, K.switch(self.cond, ag * 0, ag + g)))
        # inheriting updates of original optimizer
        self.updates.extend(self.optimizer.get_updates(loss, params)[1:])
        self.weights.extend(self.optimizer.weights)
        return self.updates
    def get_config(self):
        iterations = K.eval(self.iterations)
        K.set_value(self.iterations, 0)
        config = self.optimizer.get_config()
        K.set_value(self.iterations, iterations)
        return config




#%% efficientnet_stage4_12layer

def efficeint_block_same_channel(x,expand,squeeze,block_name):
    with tf.name_scope(block_name):
        m=tf.keras.layers.Conv2D(expand,(1,1), padding= "same")(x)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3),strides=(1, 1), padding="same")(m)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.Conv2D(squeeze,(1,1), padding= "same")(m)
        m=tf.keras.layers.BatchNormalization()(m)

        return tf.keras.layers.Add()([m,x])


def efficeint_block_different_channel(x,expand,squeeze,block_name):
    with tf.name_scope(block_name):
        m=tf.keras.layers.Conv2D(expand,(1,1),padding= "same")(x)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3),strides=(1, 1),padding="same")(m)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.Conv2D(squeeze,(1,1),padding= "same")(m)
        m=tf.keras.layers.BatchNormalization()(m)

        z=tf.keras.layers.Conv2D(squeeze,(1,1),padding= "same")(x)
        z=tf.keras.layers.BatchNormalization()(z)

        return tf.keras.layers.Add()([m,z])

model_input = Input(shape=(32,32,3))
#rotation=tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)(model_input)
#flip=tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(rotation)

conv1=tf.keras.layers.Conv2D(128,(3,3), padding="same")(model_input)
batch_norm=tf.keras.layers.BatchNormalization()(conv1)
relu=tf.keras.layers.ReLU()(batch_norm)

stage1_block1=efficeint_block_same_channel(relu,128,128,"stage1_block1")
stage1_block2=efficeint_block_same_channel(stage1_block1,768,128,"stage1_block2")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage1_block2)


stage2_block1=efficeint_block_different_channel(pool1,1536,256,"stage2_block1")
stage2_block2=efficeint_block_same_channel(stage2_block1,1536,256,"stage2_block2")
stage2_block3=efficeint_block_same_channel(stage2_block2,1536,256,"stage2_block3")
stage2_block4=efficeint_block_same_channel(stage2_block3,1536,256,"stage2_block4")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage2_block4)


stage3_block1=efficeint_block_different_channel(pool1,3072,512,"stage3_block1")
stage3_block2=efficeint_block_same_channel(stage3_block1,3072,512,"stage3_block2")
stage3_block3=efficeint_block_same_channel(stage3_block2,3072,512,"stage3_block3")
stage3_block4=efficeint_block_same_channel(stage3_block3,3072,512,"stage3_block4")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage3_block4)


stage4_block1=efficeint_block_different_channel(pool1,6144,1024,"stage4_block1")
stage4_block2=efficeint_block_same_channel(stage4_block1,6144,1024,"stage4_block2")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage4_block2)

flatten1= tf.keras.layers.Flatten()(pool1)

dense1=tf.keras.layers.Dense(1024, activation='relu',kernel_initializer='random_normal'
            )(flatten1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense1=tf.keras.layers.Dense(512, activation='relu',kernel_initializer='random_normal'
            )(dense1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense3=tf.keras.layers.Dense(100, activation='softmax',kernel_initializer='random_normal'
 )(dense1)


model = Model(inputs=model_input, outputs=dense3)

loss_fn=tf.keras.losses.CategoricalCrossentropy()

from tensorflow.keras.optimizers import Adam

opt=Adam(
    learning_rate=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam",
    
    )


plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer=opt,
              loss=loss_fn,
              metrics=['accuracy'])
print(model.summary())


#%%
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#early stopping to monitor the validation loss and avoid overfitting
early_stop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=30, restore_best_weights=True)

#reducing learning rate on plateau
rlrop = ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience= 10, factor= 0.5, min_lr= 1e-8, verbose=1)
#X_train, X_validation, y_train, y_validation = train_test_split(x_train, y_train_matrix, test_size=0.2, random_state=93)

train_datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True
)

model.fit(x=train_datagen.flow(x_train, y_train_matrix,batch_size=25),
  validation_data = (x_test,y_test_matrix),epochs=300,verbose=1,batch_size=50,

  shuffle=True, use_multiprocessing=False,workers=6,callbacks = [early_stop, rlrop])




# %%




def efficeint_block_same_channel(x,expand,squeeze,block_name):
    with tf.name_scope(block_name):
        m=tf.keras.layers.Conv2D(expand,(1,1), padding= "same")(x)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3),strides=(1, 1), padding="same")(m)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.Conv2D(squeeze,(1,1), padding= "same")(m)
        m=tf.keras.layers.BatchNormalization()(m)

        return tf.keras.layers.Add()([m,x])


def efficeint_block_different_channel(x,expand,squeeze,block_name):
    with tf.name_scope(block_name):
        m=tf.keras.layers.Conv2D(expand,(1,1),padding= "same")(x)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3),strides=(1, 1),padding="same")(m)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.Conv2D(squeeze,(1,1),padding= "same")(m)
        m=tf.keras.layers.BatchNormalization()(m)

        z=tf.keras.layers.Conv2D(squeeze,(1,1),padding= "same")(x)
        z=tf.keras.layers.BatchNormalization()(z)

        return tf.keras.layers.Add()([m,z])

model_input = Input(shape=(32,32,3))
#rotation=tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)(model_input)
#flip=tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(rotation)

conv1=tf.keras.layers.Conv2D(128,(3,3), padding="same")(model_input)
batch_norm=tf.keras.layers.BatchNormalization()(conv1)
relu=tf.keras.layers.ReLU()(batch_norm)

stage1_block1=efficeint_block_same_channel(relu,128,128,"stage1_block1")
stage1_block2=efficeint_block_same_channel(stage1_block1,768,128,"stage1_block2")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage1_block2)


stage2_block1=efficeint_block_different_channel(pool1,1536,256,"stage2_block1")
stage2_block2=efficeint_block_same_channel(stage2_block1,1536,256,"stage2_block2")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage2_block2)


stage3_block1=efficeint_block_different_channel(pool1,3072,512,"stage3_block1")
stage3_block2=efficeint_block_same_channel(stage3_block1,3072,512,"stage3_block2")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage3_block2)


stage4_block1=efficeint_block_different_channel(pool1,6144,1024,"stage4_block1")
stage4_block2=efficeint_block_same_channel(stage4_block1,6144,1024,"stage4_block2")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage4_block2)

flatten1= tf.keras.layers.Flatten()(pool1)

dense1=tf.keras.layers.Dense(1024, activation='relu',kernel_initializer='random_normal'
            )(flatten1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense1=tf.keras.layers.Dense(512, activation='relu',kernel_initializer='random_normal'
            )(dense1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense3=tf.keras.layers.Dense(100, activation='softmax',kernel_initializer='random_normal'
 )(dense1)

 
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

#opt = AdamAccumulate(lr=0.0001, decay=1e-5, accum_iters=5)


plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer=adam,
              loss=loss_fn,
              metrics=['accuracy'])
print(model.summary())






#%%  efficientnet_stage4_16layer

def efficeint_block_same_channel(x,expand,squeeze,block_name):
    with tf.name_scope(block_name):
        m=tf.keras.layers.Conv2D(expand,(1,1), padding= "same")(x)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3),strides=(1, 1), padding="same")(m)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.Conv2D(squeeze,(1,1), padding= "same")(m)
        m=tf.keras.layers.BatchNormalization()(m)

        return tf.keras.layers.Add()([m,x])


def efficeint_block_different_channel(x,expand,squeeze,block_name):
    with tf.name_scope(block_name):
        m=tf.keras.layers.Conv2D(expand,(1,1),padding= "same")(x)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3),strides=(1, 1),padding="same")(m)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.Conv2D(squeeze,(1,1),padding= "same")(m)
        m=tf.keras.layers.BatchNormalization()(m)

        z=tf.keras.layers.Conv2D(squeeze,(1,1),padding= "same")(x)
        z=tf.keras.layers.BatchNormalization()(z)

        return tf.keras.layers.Add()([m,z])

model_input = Input(shape=(32,32,3))
#rotation=tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)(model_input)
#flip=tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(rotation)

conv1=tf.keras.layers.Conv2D(128,(3,3), padding="same")(model_input)
batch_norm=tf.keras.layers.BatchNormalization()(conv1)
relu=tf.keras.layers.ReLU()(batch_norm)

stage1_block1=efficeint_block_same_channel(relu,128,128,"stage1_block1")
stage1_block2=efficeint_block_same_channel(stage1_block1,768,128,"stage1_block2")
stage1_block2=efficeint_block_same_channel(stage1_block2,768,128,"stage1_block3")
stage1_block2=efficeint_block_same_channel(stage1_block2,768,128,"stage1_block4")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage1_block2)


stage2_block1=efficeint_block_different_channel(pool1,1536,256,"stage2_block1")
stage2_block2=efficeint_block_same_channel(stage2_block1,1536,256,"stage2_block2")
stage2_block3=efficeint_block_same_channel(stage2_block2,1536,256,"stage2_block3")
stage2_block4=efficeint_block_same_channel(stage2_block3,1536,256,"stage2_block4")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage2_block4)


stage3_block1=efficeint_block_different_channel(pool1,3072,512,"stage3_block1")
stage3_block2=efficeint_block_same_channel(stage3_block1,3072,512,"stage3_block2")
stage3_block3=efficeint_block_same_channel(stage3_block2,3072,512,"stage3_block3")
stage3_block4=efficeint_block_same_channel(stage3_block3,3072,512,"stage3_block4")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage3_block4)


stage4_block1=efficeint_block_different_channel(pool1,6144,1024,"stage4_block1")
stage4_block2=efficeint_block_same_channel(stage4_block1,6144,1024,"stage4_block2")
stage4_block2=efficeint_block_same_channel(stage4_block2,6144,1024,"stage4_block3")
stage4_block2=efficeint_block_same_channel(stage4_block2,6144,1024,"stage4_block4")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage4_block2)

flatten1= tf.keras.layers.Flatten()(pool1)

dense1=tf.keras.layers.Dense(1024, activation='relu',kernel_initializer='random_normal'
            )(flatten1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense1=tf.keras.layers.Dense(512, activation='relu',kernel_initializer='random_normal'
            )(dense1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense3=tf.keras.layers.Dense(100, activation='softmax',kernel_initializer='random_normal'
 )(dense1)


model = Model(inputs=model_input, outputs=dense3)
#mixed_precision.set_global_policy('mixed_float16')
#model = GAModelWrapper(accum_steps=5, mixed_precision=False, inputs=model.input, outputs=model.output)


loss_fn=tf.keras.losses.CategoricalCrossentropy()

from tensorflow.keras.optimizers import Adam

opt=Adam(
    learning_rate=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam",
    
    )
#opt = mixed_precision.LossScaleOptimizer(opt)

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer=opt,
              loss=loss_fn,
              metrics=['accuracy'])
print(model.summary())
# %%


def efficeint_block_same_channel(x,expand,squeeze,block_name):
    with tf.name_scope(block_name):
        m=tf.keras.layers.Conv2D(expand,(1,1), padding= "same")(x)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3),strides=(1, 1), padding="same")(m)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.Conv2D(squeeze,(1,1), padding= "same")(m)
        m=tf.keras.layers.BatchNormalization()(m)

        return tf.keras.layers.Add()([m,x])


def efficeint_block_different_channel(x,expand,squeeze,block_name):
    with tf.name_scope(block_name):
        m=tf.keras.layers.Conv2D(expand,(1,1),padding= "same")(x)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3),strides=(1, 1),padding="same")(m)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.Conv2D(squeeze,(1,1),padding= "same")(m)
        m=tf.keras.layers.BatchNormalization()(m)

        z=tf.keras.layers.Conv2D(squeeze,(1,1),padding= "same")(x)
        z=tf.keras.layers.BatchNormalization()(z)

        return tf.keras.layers.Add()([m,z])

model_input = Input(shape=(32,32,3))
#rotation=tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)(model_input)
#flip=tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(rotation)

conv1=tf.keras.layers.Conv2D(64,(3,3), padding="same")(model_input)
batch_norm=tf.keras.layers.BatchNormalization()(conv1)
relu=tf.keras.layers.ReLU()(batch_norm)

stage1_block1=efficeint_block_same_channel(relu,64,64,"stage1_block1")
stage1_block2=efficeint_block_same_channel(stage1_block1,384,64,"stage1_block2")
stage1_block2=efficeint_block_same_channel(stage1_block2,384,64,"stage1_block3")
stage1_block2=efficeint_block_same_channel(stage1_block2,384,64,"stage1_block4")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage1_block2)


stage2_block1=efficeint_block_different_channel(pool1,768,128,"stage2_block1")
stage2_block2=efficeint_block_same_channel(stage2_block1,768,128,"stage2_block2")
stage2_block3=efficeint_block_same_channel(stage2_block2,768,128,"stage2_block3")
stage2_block4=efficeint_block_same_channel(stage2_block3,768,128,"stage2_block4")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage2_block4)


stage3_block1=efficeint_block_different_channel(pool1,1536,256,"stage3_block1")
stage3_block2=efficeint_block_same_channel(stage3_block1,1536,256,"stage3_block2")
stage3_block3=efficeint_block_same_channel(stage3_block2,1536,256,"stage3_block3")
stage3_block4=efficeint_block_same_channel(stage3_block3,1536,256,"stage3_block4")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage3_block4)


stage4_block1=efficeint_block_different_channel(pool1,3072,512,"stage4_block1")
stage4_block2=efficeint_block_same_channel(stage4_block1,3072,512,"stage4_block2")
stage4_block2=efficeint_block_same_channel(stage4_block2,3072,512,"stage4_block3")
stage4_block2=efficeint_block_same_channel(stage4_block2,3072,512,"stage4_block4")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage4_block2)

flatten1= tf.keras.layers.Flatten()(pool1)

dense1=tf.keras.layers.Dense(1024, activation='relu',kernel_initializer='random_normal'
            )(flatten1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense1=tf.keras.layers.Dense(512, activation='relu',kernel_initializer='random_normal'
            )(dense1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense3=tf.keras.layers.Dense(100, activation='softmax',kernel_initializer='random_normal'
 )(dense1)


model = Model(inputs=model_input, outputs=dense3)
#mixed_precision.set_global_policy('mixed_float16')
#model = GAModelWrapper(accum_steps=5, mixed_precision=False, inputs=model.input, outputs=model.output)


loss_fn=tf.keras.losses.CategoricalCrossentropy()

from tensorflow.keras.optimizers import Adam

opt=Adam(
    learning_rate=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam",
    
    )


plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer=opt,
              loss=loss_fn,
              metrics=['accuracy'])
print(model.summary())


#%%

def efficeint_block_same_channel(x,expand,squeeze,block_name):
    with tf.name_scope(block_name):
        m=tf.keras.layers.Conv2D(expand,(1,1), padding= "same")(x)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3),strides=(1, 1), padding="same")(m)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)

        t=tf.keras.layers.GlobalAveragePooling2D()(m)
        t=tf.keras.layers.Reshape((1,1,expand))(t)
        t=tf.keras.layers.Conv2D(squeeze/4,(1,1), padding= "same")(t)
        t=tf.keras.layers.Conv2D(expand,(1,1), padding= "same")(t)

        m=tf.keras.layers.Multiply()([m,t])
        m=tf.keras.layers.Conv2D(squeeze,(1,1), padding= "same")(m)
        m=tf.keras.layers.BatchNormalization()(m)

        return tf.keras.layers.Add()([m,x])


def efficeint_block_different_channel(x,expand,squeeze,block_name):
    with tf.name_scope(block_name):
        m=tf.keras.layers.Conv2D(expand,(1,1),padding= "same")(x)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3),strides=(1, 1),padding="same")(m)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        t=tf.keras.layers.GlobalAveragePooling2D()(m)
        t=tf.keras.layers.Reshape((1,1,expand))(t)
        t=tf.keras.layers.Conv2D(squeeze/4,(1,1), padding= "same")(t)
        t=tf.keras.layers.Conv2D(expand,(1,1), padding= "same")(t)

        m=tf.keras.layers.Multiply()([m,t])



        m=tf.keras.layers.Conv2D(squeeze,(1,1),padding= "same")(m)
        m=tf.keras.layers.BatchNormalization()(m)

        z=tf.keras.layers.Conv2D(squeeze,(1,1),padding= "same")(x)
        z=tf.keras.layers.BatchNormalization()(z)

        return tf.keras.layers.Add()([m,z])
model_input = Input(shape=(32,32,3))
#rotation=tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)(model_input)
#flip=tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(rotation)

conv1=tf.keras.layers.Conv2D(64,(3,3), padding="same")(model_input)
batch_norm=tf.keras.layers.BatchNormalization()(conv1)
relu=tf.keras.layers.ReLU()(batch_norm)

stage1_block1=efficeint_block_same_channel(relu,64,64,"stage1_block1")
stage1_block2=efficeint_block_same_channel(stage1_block1,384,64,"stage1_block2")
stage1_block2=efficeint_block_same_channel(stage1_block2,384,64,"stage1_block3")
stage1_block2=efficeint_block_same_channel(stage1_block2,384,64,"stage1_block4")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage1_block2)


stage2_block1=efficeint_block_different_channel(pool1,768,128,"stage2_block1")
stage2_block2=efficeint_block_same_channel(stage2_block1,768,128,"stage2_block2")
stage2_block3=efficeint_block_same_channel(stage2_block2,768,128,"stage2_block3")
stage2_block4=efficeint_block_same_channel(stage2_block3,768,128,"stage2_block4")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage2_block4)


stage3_block1=efficeint_block_different_channel(pool1,1536,256,"stage3_block1")
stage3_block2=efficeint_block_same_channel(stage3_block1,1536,256,"stage3_block2")
stage3_block3=efficeint_block_same_channel(stage3_block2,1536,256,"stage3_block3")
stage3_block4=efficeint_block_same_channel(stage3_block3,1536,256,"stage3_block4")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage3_block4)


stage4_block1=efficeint_block_different_channel(pool1,3072,512,"stage4_block1")
stage4_block2=efficeint_block_same_channel(stage4_block1,3072,512,"stage4_block2")
stage4_block2=efficeint_block_same_channel(stage4_block2,3072,512,"stage4_block3")
stage4_block2=efficeint_block_same_channel(stage4_block2,3072,512,"stage4_block4")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage4_block2)

flatten1= tf.keras.layers.Flatten()(pool1)

dense1=tf.keras.layers.Dense(1024, activation='relu',kernel_initializer='random_normal'
            )(flatten1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense1=tf.keras.layers.Dense(512, activation='relu',kernel_initializer='random_normal'
            )(dense1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense3=tf.keras.layers.Dense(100, activation='softmax',kernel_initializer='random_normal'
 )(dense1)


model = Model(inputs=model_input, outputs=dense3)
#mixed_precision.set_global_policy('mixed_float16')
#model = GAModelWrapper(accum_steps=5, mixed_precision=False, inputs=model.input, outputs=model.output)


loss_fn=tf.keras.losses.CategoricalCrossentropy()

from tensorflow.keras.optimizers import Adam

opt=Adam(
    learning_rate=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam",
    
    )


plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer=opt,
              loss=loss_fn,
              metrics=['accuracy'])
print(model.summary())
#%% efficientnet_stage4_16layer_se

def efficeint_block_same_channel(x,expand,squeeze,block_name):
    with tf.name_scope(block_name):
        m=tf.keras.layers.Conv2D(expand,(1,1), padding= "same")(x)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3),strides=(1, 1), padding="same")(m)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)

        t=tf.keras.layers.GlobalAveragePooling2D()(m)
        t=tf.keras.layers.Reshape((1,1,expand))(t)
        t=tf.keras.layers.Conv2D(squeeze/4,(1,1), padding= "same")(t)
        t=tf.keras.layers.Conv2D(expand,(1,1), padding= "same")(t)

        m=tf.keras.layers.Multiply()([m,t])
        m=tf.keras.layers.Conv2D(squeeze,(1,1), padding= "same")(m)
        m=tf.keras.layers.BatchNormalization()(m)

        return tf.keras.layers.Add()([m,x])


def efficeint_block_different_channel(x,expand,squeeze,block_name):
    with tf.name_scope(block_name):
        m=tf.keras.layers.Conv2D(expand,(1,1),padding= "same")(x)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3),strides=(1, 1),padding="same")(m)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        t=tf.keras.layers.GlobalAveragePooling2D()(m)
        t=tf.keras.layers.Reshape((1,1,expand))(t)
        t=tf.keras.layers.Conv2D(squeeze/4,(1,1), padding= "same")(t)
        t=tf.keras.layers.Conv2D(expand,(1,1), padding= "same")(t)

        m=tf.keras.layers.Multiply()([m,t])



        m=tf.keras.layers.Conv2D(squeeze,(1,1),padding= "same")(m)
        m=tf.keras.layers.BatchNormalization()(m)

        z=tf.keras.layers.Conv2D(squeeze,(1,1),padding= "same")(x)
        z=tf.keras.layers.BatchNormalization()(z)

        return tf.keras.layers.Add()([m,z])


model_input = Input(shape=(32,32,3))
#rotation=tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)(model_input)
#flip=tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(rotation)

conv1=tf.keras.layers.Conv2D(128,(3,3), padding="same")(model_input)
batch_norm=tf.keras.layers.BatchNormalization()(conv1)
relu=tf.keras.layers.ReLU()(batch_norm)

stage1_block1=efficeint_block_same_channel(relu,128,128,"stage1_block1")
stage1_block2=efficeint_block_same_channel(stage1_block1,768,128,"stage1_block2")
stage1_block2=efficeint_block_same_channel(stage1_block2,768,128,"stage1_block3")
stage1_block2=efficeint_block_same_channel(stage1_block2,768,128,"stage1_block4")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage1_block2)


stage2_block1=efficeint_block_different_channel(pool1,1536,256,"stage2_block1")
stage2_block2=efficeint_block_same_channel(stage2_block1,1536,256,"stage2_block2")
stage2_block3=efficeint_block_same_channel(stage2_block2,1536,256,"stage2_block3")
stage2_block4=efficeint_block_same_channel(stage2_block3,1536,256,"stage2_block4")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage2_block4)


stage3_block1=efficeint_block_different_channel(pool1,3072,512,"stage3_block1")
stage3_block2=efficeint_block_same_channel(stage3_block1,3072,512,"stage3_block2")
stage3_block3=efficeint_block_same_channel(stage3_block2,3072,512,"stage3_block3")
stage3_block4=efficeint_block_same_channel(stage3_block3,3072,512,"stage3_block4")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage3_block4)


stage4_block1=efficeint_block_different_channel(pool1,6144,1024,"stage4_block1")
stage4_block2=efficeint_block_same_channel(stage4_block1,6144,1024,"stage4_block2")
stage4_block2=efficeint_block_same_channel(stage4_block2,6144,1024,"stage4_block3")
stage4_block2=efficeint_block_same_channel(stage4_block2,6144,1024,"stage4_block4")
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(stage4_block2)

flatten1= tf.keras.layers.Flatten()(pool1)

dense1=tf.keras.layers.Dense(1024, activation='relu',kernel_initializer='random_normal'
            )(flatten1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense1=tf.keras.layers.Dense(512, activation='relu',kernel_initializer='random_normal'
            )(dense1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense3=tf.keras.layers.Dense(100, activation='softmax',kernel_initializer='random_normal'
 )(dense1)


model = Model(inputs=model_input, outputs=dense3)
#mixed_precision.set_global_policy('mixed_float16')
#model = GAModelWrapper(accum_steps=5, mixed_precision=False, inputs=model.input, outputs=model.output)


loss_fn=tf.keras.losses.CategoricalCrossentropy()

from tensorflow.keras.optimizers import Adam

opt=Adam(
    learning_rate=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam",
    
    )
#opt = mixed_precision.LossScaleOptimizer(opt)

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer=opt,
              loss=loss_fn,
              metrics=['accuracy'])
print(model.summary())
# %%



def efficeint_block_same_channel(x,expand,squeeze,block_name,firststride):
    with tf.name_scope(block_name):
        m=tf.keras.layers.Conv2D(expand,(1,1), padding= "same")(x)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3),strides=(firststride, firststride), padding="same")(m)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)

        t=tf.keras.layers.GlobalAveragePooling2D()(m)
        t=tf.keras.layers.Reshape((1,1,expand))(t)
        t=tf.keras.layers.Conv2D(squeeze/6,(1,1), padding= "same", activation='relu')(t)
        t=tf.keras.layers.Conv2D(expand,(1,1), padding= "same", activation='sigmoid')(t)

        m=tf.keras.layers.Multiply()([m,t])
        m=tf.keras.layers.Conv2D(squeeze,(1,1), padding= "same")(m)
        m=tf.keras.layers.BatchNormalization()(m)

        return tf.keras.layers.Add()([m,x])


def efficeint_block_different_channel(x,expand,squeeze,block_name,firststride):
    with tf.name_scope(block_name):
        m=tf.keras.layers.Conv2D(expand,(1,1),padding= "same")(x)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        m=tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3),strides=(firststride, firststride),padding="same")(m)
        m=tf.keras.layers.BatchNormalization()(m)
        m=tf.keras.layers.ReLU()(m)
        t=tf.keras.layers.GlobalAveragePooling2D()(m)
        t=tf.keras.layers.Reshape((1,1,expand))(t)
        t=tf.keras.layers.Conv2D(squeeze/6,(1,1), padding= "same", activation='relu')(t)
        t=tf.keras.layers.Conv2D(expand,(1,1), padding= "same", activation='sigmoid')(t)

        m=tf.keras.layers.Multiply()([m,t])



        m=tf.keras.layers.Conv2D(squeeze,(1,1),padding= "same")(m)
        m=tf.keras.layers.BatchNormalization()(m)

        #z=tf.keras.layers.Conv2D(squeeze,(1,1),padding= "same")(x)
        #z=tf.keras.layers.BatchNormalization()(z)

        return tf.keras.layers.Add()([m])


def mbconv6(x,expand,squeeze,blocknumber,firststrides,inputchannel):
    for i in range(blocknumber):
      if i==0:
          if inputchannel != squeeze:
              m=efficeint_block_different_channel(x,expand,squeeze,"block",firststrides)
          else:
              m=efficeint_block_same_channel(x,expand,squeeze,"block",firststrides)
      else:
          m=efficeint_block_same_channel(m,expand,squeeze,"block",1)

    return m

model_input = Input(shape=(32,32,3))
#rotation=tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)(model_input)
#flip=tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(rotation)

conv1=tf.keras.layers.Conv2D(32,(3,3),strides=(2,2), padding="same")(model_input)
batch_norm=tf.keras.layers.BatchNormalization()(conv1)
relu=tf.keras.layers.ReLU()(batch_norm)

stage1=mbconv6(relu,expand=16,squeeze=16,blocknumber=1,firststrides=1,inputchannel=32)


stage2=mbconv6(stage1,expand=144,squeeze=24,blocknumber=2,firststrides=2,inputchannel=16)

stage3=mbconv6(stage2,expand=240,squeeze=40,blocknumber=2,firststrides=1,inputchannel=24)


stage4=mbconv6(stage3,expand=480,squeeze=80,blocknumber=3,firststrides=2,inputchannel=40)


stage5=mbconv6(stage4,expand=112*6,squeeze=112,blocknumber=3,firststrides=1,inputchannel=80)

stage6=mbconv6(stage5,expand=192*6,squeeze=192,blocknumber=4,firststrides=2,inputchannel=112)

stage7=mbconv6(stage6,expand=320*6,squeeze=320,blocknumber=1,firststrides=1,inputchannel=192)


conv1=tf.keras.layers.Conv2D(1280,(1,1),strides=(1,1), padding="same")(stage7)
batch_norm=tf.keras.layers.BatchNormalization()(conv1)
pool1=tf.keras.layers.GlobalAveragePooling2D()(batch_norm)


dense1=tf.keras.layers.Dropout(0.2)(pool1)
dense3=tf.keras.layers.Dense(100, activation='softmax',kernel_initializer='random_normal'
 )(dense1)


model = Model(inputs=model_input, outputs=dense3)
#mixed_precision.set_global_policy('mixed_float16')
#model = GAModelWrapper(accum_steps=5, mixed_precision=False, inputs=model.input, outputs=model.output)


loss_fn=tf.keras.losses.CategoricalCrossentropy()

from tensorflow.keras.optimizers import Adam

opt=Adam(
    learning_rate=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam",
    
    )
#opt = mixed_precision.LossScaleOptimizer(opt)

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer=opt,
              loss=loss_fn,
              metrics=['accuracy'])
print(model.summary())   
# %%
