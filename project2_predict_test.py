#%%
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
# 在开启对话session前，先创建一个 tf.ConfigProto() 实例对象

gpuConfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
gpuConfig.gpu_options.allow_growth=True



# 把你的配置部署到session  变量名 sess 无所谓
sess1 =tf.compat.v1.Session(config=gpuConfig)


#%% input training data

import os
from os import listdir
from os.path import isfile, join

import cv2

label_folder = []
total_size = 0
data_path = "/mnt/c/Users/ryan7/Downloads/Training_data"



for root, dirts, files in os.walk(data_path):
    total_size += len(files)
    for dirt in dirts:
        label_folder.append(dirt)
        #total_size = total_size+  len(files)

print("found", total_size, "files.")
print("folder:", label_folder)

# %% load img train
import numpy as np

base_x_train = []
base_y_train = []

inputshape=[32,32,3]
outputclass=30

for i in range(len(label_folder)):
    labelPath = data_path + r"/" + label_folder[i]

    FileName = [f for f in listdir(labelPath) if isfile(join(labelPath, f))]

    for j in range(len(FileName)):
        path = labelPath + r"/" + FileName[j]

        img = cv2.imread(path, cv2.IMREAD_COLOR)

        base_x_train.append(img)
        base_y_train.append(label_folder[i])

print(np.array(base_x_train).shape)
print(np.array(base_y_train).shape)

#%% to categorical

from tensorflow.keras.utils import to_categorical

base_y_train=to_categorical(base_y_train)

print(np.array(base_x_train).shape)
print(np.array(base_y_train).shape)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    np.array(base_x_train)/255, np.array(base_y_train), test_size=0.05
)


#%%

import random
import matplotlib.pyplot as plt

idx= random.randint(0,x_train.shape[0])
plt.imshow(x_train[idx])
plt.show()

print("Ans:",np.argmax(y_train[idx]))
print("Ans(one-hot):",y_train[idx])





#%% model densenet_stage4_16layer


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

model_input = Input(shape=inputshape)
#rotation=tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)(model_input)
#flip=tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(rotation)

conv1=tf.keras.layers.Conv2D(128,(3,3), padding="same")(model_input)
batch_norm=tf.keras.layers.BatchNormalization()(conv1)
relu=tf.keras.layers.ReLU()(batch_norm)
t=tf.keras.layers.Conv2D(128,(1,1),padding= "same")(relu)

stage1_block1=efficeint_block_same_channel(t,128,128,"stage1_block1")
stage1_block2=efficeint_block_same_channel(stage1_block1,768,128,"stage1_block2")
#stage1_block2=efficeint_block_same_channel(stage1_block2,768,128,"stage1_block3")
#stage1_block2=efficeint_block_same_channel(stage1_block2,768,128,"stage1_block4")

s=tf.keras.layers.Add()([stage1_block2,t])
s=tf.keras.layers.BatchNormalization()(s)
s=tf.keras.layers.ReLU()(s)
s=tf.keras.layers.Conv2D(256,(1,1),padding= "same")(s)

pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(s)


stage2_block1=efficeint_block_different_channel(pool1,1536,256,"stage2_block1")
stage2_block2=efficeint_block_same_channel(stage2_block1,1536,256,"stage2_block2")
stage2_block3=efficeint_block_same_channel(stage2_block2,1536,256,"stage2_block3")
stage2_block4=efficeint_block_same_channel(stage2_block3,1536,256,"stage2_block4")

h=tf.keras.layers.Conv2D(256,(1,1),padding= "same")(t)
h=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(h)

s=tf.keras.layers.Add()([pool1,stage2_block4,h])
s=tf.keras.layers.BatchNormalization()(s)
s=tf.keras.layers.ReLU()(s)
s=tf.keras.layers.Conv2D(512,(1,1),padding= "same")(s)

pool2=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(s)


stage3_block1=efficeint_block_different_channel(pool2,3072,512,"stage3_block1")
stage3_block2=efficeint_block_same_channel(stage3_block1,3072,512,"stage3_block2")
stage3_block3=efficeint_block_same_channel(stage3_block2,3072,512,"stage3_block3")
stage3_block4=efficeint_block_same_channel(stage3_block3,3072,512,"stage3_block4")

h=tf.keras.layers.Conv2D(512,(1,1),padding= "same")(h)
h=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(h)

a=tf.keras.layers.Conv2D(512,(1,1),padding= "same")(pool1)
a=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(a)

s=tf.keras.layers.Add()([h,stage3_block4,pool2,a])
s=tf.keras.layers.BatchNormalization()(s)
s=tf.keras.layers.ReLU()(s)
s=tf.keras.layers.Conv2D(1024,(1,1),padding= "same")(s)


pool3=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(s)


stage4_block1=efficeint_block_different_channel(pool3,6144,1024,"stage4_block1")
stage4_block2=efficeint_block_same_channel(stage4_block1,6144,1024,"stage4_block2")
#stage4_block2=efficeint_block_same_channel(stage4_block2,6144,1024,"stage4_block3")
#stage4_block2=efficeint_block_same_channel(stage4_block2,6144,1024,"stage4_block4")

h=tf.keras.layers.Conv2D(1024,(1,1),padding= "same")(h)
h=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(h)

a=tf.keras.layers.Conv2D(1024,(1,1),padding= "same")(a)
a=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(a)

b=tf.keras.layers.Conv2D(1024,(1,1),padding= "same")(pool2)
b=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(b)

s=tf.keras.layers.Add()([h,stage4_block2,pool3,a,b])
s=tf.keras.layers.BatchNormalization()(s)
s=tf.keras.layers.ReLU()(s)
s=tf.keras.layers.Conv2D(1024,(1,1),padding= "same")(s)


pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(s)

flatten1= tf.keras.layers.Flatten()(pool1)

dense1=tf.keras.layers.Dense(1024, activation='relu',kernel_initializer='random_normal'
            )(flatten1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense1=tf.keras.layers.Dense(256, activation='relu',kernel_initializer='random_normal'
            )(dense1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense1=tf.keras.layers.Dense(64, activation='relu',kernel_initializer='random_normal'
            )(dense1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense3=tf.keras.layers.Dense(outputclass, activation='softmax',kernel_initializer='random_normal'
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


 #%%


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

model_input = Input(shape=inputshape)
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

dense1=tf.keras.layers.Dense(1024, activation='swish',kernel_initializer='random_normal'
            )(flatten1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense1=tf.keras.layers.Dense(256, activation='swish',kernel_initializer='random_normal'
            )(dense1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense1=tf.keras.layers.Dense(64, activation='swish',kernel_initializer='random_normal'
            )(dense1)
dense1=tf.keras.layers.Dropout(0.2)(dense1)
dense3=tf.keras.layers.Dense(outputclass, activation='softmax',kernel_initializer='random_normal'
 )(dense1)


model = Model(inputs=model_input, outputs=dense3)
#mixed_precision.set_global_policy('mixed_float16')
#model = GAModelWrapper(accum_steps=5, mixed_precision=False, inputs=model.input, outputs=model.output)


loss_fn=tf.keras.losses.CategoricalCrossentropy()

from tensorflow.keras.optimizers import Adam

opt=Adam(
    learning_rate=0.0001,
    
    
    )
#opt = mixed_precision.LossScaleOptimizer(opt)

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer=opt,
              loss=loss_fn,
              metrics=['accuracy'])
print(model.summary())




#%% efficientnet b0
import math

dropoutrate=0.1
multiplier_fil=2.0

multiplier_repeat=2.0


def round_filters(filters,multiplier):
  depth_divisor=8
  min_depth=None
  min_depth=min_depth or depth_divisor
  filters=filters*multiplier
  new_filters=max(min_depth,int(filters+depth_divisor/2)//depth_divisor*depth_divisor)
  if new_filters<0.9*filters:
      new_filters+=depth_divisor
  
  return int(new_filters)

def round_repeats(repeats,multiplier):
  if not multiplier:
    return repeats
  return int(math.ceil(multiplier*repeats))




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
        #m=tf.keras.layers.Dropout(dropoutrate)(m)

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

model_input = Input(shape=inputshape)
#rotation=tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)(model_input)
#flip=tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(rotation)

conv1=tf.keras.layers.Conv2D(round_filters(32,multiplier_fil),(3,3),strides=(2,2), padding="same")(model_input)
batch_norm=tf.keras.layers.BatchNormalization()(conv1)
relu=tf.keras.layers.ReLU()(batch_norm)

stage1_fil=round_filters(16,multiplier_fil)
stage2_fil=round_filters(24,multiplier_fil)
stage3_fil=round_filters(40,multiplier_fil)
stage4_fil=round_filters(80,multiplier_fil)
stage5_fil=round_filters(112,multiplier_fil)
stage6_fil=round_filters(192,multiplier_fil)
stage7_fil=round_filters(320,multiplier_fil)

stage1=mbconv6(relu,expand=stage1_fil,squeeze=stage1_fil,blocknumber=round_repeats(1,multiplier_repeat),firststrides=1,inputchannel=round_filters(32,multiplier_fil))


stage2=mbconv6(stage1,expand=stage2_fil*6,squeeze=stage2_fil,blocknumber=round_repeats(2,multiplier_repeat),firststrides=2,inputchannel=stage1_fil)

stage3=mbconv6(stage2,expand=stage3_fil*6,squeeze=stage3_fil,blocknumber=round_repeats(2,multiplier_repeat),firststrides=1,inputchannel=stage2_fil)


stage4=mbconv6(stage3,expand=stage4_fil*6,squeeze=stage4_fil,blocknumber=round_repeats(3,multiplier_repeat),firststrides=2,inputchannel=stage3_fil)


stage5=mbconv6(stage4,expand=stage5_fil*6,squeeze=stage5_fil,blocknumber=round_repeats(3,multiplier_repeat),firststrides=1,inputchannel=stage4_fil)

stage6=mbconv6(stage5,expand=stage6_fil*6,squeeze=stage6_fil,blocknumber=round_repeats(4,multiplier_repeat),firststrides=2,inputchannel=stage5_fil)

stage7=mbconv6(stage6,expand=stage7_fil*6,squeeze=stage7_fil,blocknumber=round_repeats(1,multiplier_repeat),firststrides=1,inputchannel=stage6_fil)


conv1=tf.keras.layers.Conv2D(round_filters(1280,multiplier_fil),(1,1),strides=(1,1), padding="same")(stage7)
batch_norm=tf.keras.layers.BatchNormalization()(conv1)
pool1=tf.keras.layers.GlobalAveragePooling2D()(batch_norm)


dense1=tf.keras.layers.Dropout(dropoutrate)(pool1)

dense1=tf.keras.layers.Dense(128, activation='relu',kernel_initializer='random_normal'
            )(dense1)
dense1=tf.keras.layers.Dropout(dropoutrate)(dense1)

dense3=tf.keras.layers.Dense(outputclass, activation='softmax',kernel_initializer='random_normal'
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

# %% fit
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#early stopping to monitor the validation loss and avoid overfitting
early_stop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20, restore_best_weights=True)

#reducing learning rate on plateau
rlrop = ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience= 5, factor= 0.5, min_lr= 1e-10, verbose=1)
#X_train, X_validation, y_train, y_validation = train_test_split(x_train, y_train_matrix, test_size=0.2, random_state=93)

tensorboard=tf.keras.callbacks.TensorBoard(log_dir="/mnt/c/Users/ryan7/Documents/GitHub/python_ai_project2/project2_predict_test_logs")
train_datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    #vertical_flip=True,
    #shear_range=0.1,
    zoom_range=0.1
)

model.fit(x=train_datagen.flow(x_train, y_train,batch_size=25),
  validation_data = (x_test,y_test),epochs=50,verbose=1,batch_size=25,

  shuffle=True, use_multiprocessing=False,workers=6,callbacks = [early_stop, rlrop,tensorboard])
#%%
import matplotlib.pyplot as plt
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#%%

model.save("/mnt/c/Users/ryan7/Documents/GitHub/python_ai_project2/test_model")


#%% self test predict
count=0
selfpredict=model.predict(x_test)
for i in range(len(selfpredict)):
    if np.argmax(selfpredict[i])==np.argmax(y_test[i]):
        count +=1

print(count/len(y_test))



#%% load img test

data_path = "/mnt/c/Users/ryan7/Downloads/410873001"

base_x_test = []

labelPath = data_path + r"/"

FileName_test = [f for f in listdir(labelPath) if isfile(join(labelPath, f))]

for j in range(len(FileName_test)):
    path = labelPath + r"/" + FileName_test[j]

    img = cv2.imread(path, cv2.IMREAD_COLOR)

    base_x_test.append(img)


testnumber = len(base_x_test)
base_x_test=np.array(base_x_test)
base_y_test=model.predict(base_x_test/255)

pred = np.zeros((testnumber, 1))
for i in range(0, testnumber):
    pred[i] = np.argmax(base_y_test[i])


for j in range(0,len(FileName_test)):
    FileName_test[j]=FileName_test[j].removesuffix(".png")


#%%
path_to_file = "/mnt/c/Users/ryan7/Downloads/"
with open(path_to_file + "410873001.txt", "w") as g:
    for t in range(1,testnumber+1):
        for j in range(testnumber):
            if int(FileName_test[j])==t:
            
                g.write(FileName_test[j] + " " + str(int(pred[j])) + "\n")
# %%
