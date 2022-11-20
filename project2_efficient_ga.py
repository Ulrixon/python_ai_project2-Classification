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
#from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
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



#%%
from tensorflow.keras.mixed_precision import LossScaleOptimizer as lso
from tensorflow.distribute import parameter_server_strategy
from tensorflow.distribute import distribution_strategy_context as ds_context
from tensorflow.util import nest
from tensorflow.keras.models import Model as _Model


class Model1(_Model):
    def fit(self, *args, batch_size: int = 32, grad_accum_steps: int = 1, **kwargs):
        """
        Shallow wrapper of Model.fit that captures batch_size and additional kwarg: grad_accum.

        Parameters
        ----------
        batch_size : int
            same as in Model.fit
        grad_accum_steps : int
            Number of steps to split batch_size into. The `batch_size` should be divisible by `grad_accum` (defaults to 1).
        """
        if grad_accum_steps == 1:
            super().fit(*args, batch_size = batch_size, **kwargs)

        self.train_function = None
        num_workers = ds_context.get_strategy().num_replicas_in_sync
        if batch_size % (grad_accum_steps * num_workers) != 0:
            raise ValueError(f'Batch size ({batch_size}) must be divisible by the Gradient accumulation steps ({grad_accum_steps}), and the number of replicas ({num_workers}), dummy!')

        self._grad_accum_ = grad_accum_steps
        self._batch_size_ = batch_size
        self._num_workers_ = num_workers
        train_step_backup = self.train_step
        self.train_step = self._train_step_
        out = super(self).fit(*args,
                              batch_size = self._batch_size_, # TODO maybe consider validation batch size
                              **kwargs)

        del self._grad_accum_
        del self._batch_size_
        del self._num_workers_
        self.train_step = train_step_backup
        return out

    def _train_step_(self, data):
        """
        Custom training step taking into account gradient accumulation for low memory training
        """

        if len(data) == 3:
            x, y, sample_weight = data
        else:
            (x, y), sample_weight = data, None


        def slice_map(struct, start, stop): # dealing with nasty nested structures
            if struct is None:
                return None # special case for sample_weight

            return nest.map_structure(lambda x: x[start:stop], struct)



        # ---------- GRAD ACCUM STUFF ----------------------------------------------------------------------------------
        step = self._batch_size_ // self._num_workers_ // self._grad_accum_
        x_ = slice_map(x, 0, step)
        y_ = slice_map(y, 0, step)
        w_ = slice_map(sample_weight, 0, step)

        with tf.GradientTape() as tape:

            y_pred = self(x_, training = True)  # Forward pass
            loss = self.compiled_loss(y_, y_pred, sample_weight = w_, regularization_losses = self.losses)
            if isinstance(self.optimizer, lso.LossScaleOptimizer):
                loss = self.optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        gradients = [gradient * (1./self._grad_accum_) for gradient in gradients]
        self.compiled_metrics.update_state(y_, y_pred)

        i = tf.constant(step)
        def cond(i, *args):
            return i < self._batch_size_

        def body(i, grad):
            x_ = slice_map(x, i, i + step)
            y_ = slice_map(y, i, i + step)
            w_ = slice_map(sample_weight, i, i + step)

            with tf.GradientTape() as tape:
                y_pred = self(x_, training = True) # Forward pass
                loss = self.compiled_loss(y_, y_pred, sample_weight = w_, regularization_losses = self.losses)
                if isinstance(self.optimizer, lso.LossScaleOptimizer):
                    loss = self.optimizer.get_scaled_loss(loss)

            _grad = tape.gradient(loss, self.trainable_variables)
            _grad = [_g * (1./self._grad_accum_) for _g in _grad]

            grad = [g + _g for g,_g in zip(grad, _grad)]

            self.compiled_metrics.update_state(y_, y_pred)
            return [i + step, grad]

        i, gradients = tf.while_loop(cond, body, [i, gradients], parallel_iterations = 1)
        # --------------------------------------------------------------------------------------------------------------



        # ---------- STUFF FROM Model._minimize ------------------------------------------------------------------------
        aggregate_grads_outside_optimizer = (self.optimizer._HAS_AGGREGATE_GRAD and not isinstance(self.distribute_strategy.extended, parameter_server_strategy.ParameterServerStrategyExtended))

        if aggregate_grads_outside_optimizer: # TODO there might be some issues with the scaling, due to the extra accumulation steps
            gradients = self.optimizer._aggregate_gradients(zip(gradients, self.trainable_variables))

        if isinstance(self.optimizer, lso.LossScaleOptimizer):
            gradients = self.optimizer.get_unscaled_gradients(gradients)

        gradients = self.optimizer._clip_gradients(gradients)
        if self.trainable_variables:
            if aggregate_grads_outside_optimizer:
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables), experimental_aggregate_gradients = False)
            else:
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # --------------------------------------------------------------------------------------------------------------


        return {m.name: m.result() for m in self.metrics}
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
#%%
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#early stopping to monitor the validation loss and avoid overfitting
early_stop = EarlyStopping(monitor='val_accuracy', mode='min', verbose=1, patience=30, restore_best_weights=True)

#reducing learning rate on plateau
rlrop = ReduceLROnPlateau(monitor='val_accuracy', mode='min', patience= 10, factor= 0.5, min_lr= 1e-8, verbose=1)
#X_train, X_validation, y_train, y_validation = train_test_split(x_train, y_train_matrix, test_size=0.2, random_state=93)

train_datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True
)

model.fit(x=train_datagen.flow(x_train, y_train_matrix,batch_size=50),
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
