import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from models import VGG16 

import os
os.environ["CUDA_VISIBLE_DEVICES"]="4" 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)



dataset, info = tfds.load('imagenette/320px-v2', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['validation']


def preprocess(image, label):
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0  
    return image, label

train_dataset = train_dataset.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

model = VGG16()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

filepath = './imagenette_vgg.h5' 
checkpoint = ModelCheckpoint(filepath=filepath,monitor='val_accuracy',verbose=1,save_best_only=True,mode='auto')
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),cooldown=0,patience=5,min_lr=0.5e-6)
callbacks = [checkpoint, lr_reducer]        
model.fit(train_dataset, epochs=50,callbacks=callbacks)

