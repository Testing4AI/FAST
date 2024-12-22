from tensorflow import keras
import tensorflow as tf
import numpy as np
from models import Lenet5

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        


x_train, x_test, y_train, y_test = load_mnist(path)
model = Lenet5()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=64)
model.evaluate(x_test, y_test)
model.save("./lenet_mnist.h5")


