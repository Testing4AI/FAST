import os
os.environ["CUDA_VISIBLE_DEVICES"]="5" 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
        
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

vocab_size = 10000  
max_length = 200  
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size, oov_char=2)
train_data = pad_sequences(train_data, maxlen=max_length, padding=padding_type, truncating=trunc_type)
test_data = pad_sequences(test_data, maxlen=max_length, padding=padding_type, truncating=trunc_type)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_data, train_labels, epochs=5, validation_data=(test_data, test_labels), batch_size=64)
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f'\nTest accuracy: {test_acc}')
# model.save('imdb_lstm_model.h5')


