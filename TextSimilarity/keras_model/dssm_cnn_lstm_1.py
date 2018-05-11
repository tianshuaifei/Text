from __future__ import print_function


from keras.layers import Conv1D, MaxPooling1D,GlobalAveragePooling1D
import keras
from keras.layers import Input, LSTM, Dense,Embedding
from keras.models import Model
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM,Merge
from keras.layers import Conv1D, MaxPooling1D,Reshape
from keras.datasets import imdb

from MaYi import data_feature as data_provider

print("Loading data...")
train_query, train_doc, train_label = data_provider.load_train_dataset()
valid_query, valid_doc, valid_label = data_provider.load_valid_dataset()
test_query, test_doc, test_label = data_provider.load_pre_dataset()
print(len(train_query), 'train sequences')

timesteps = 30
num_classes = 2
max_features=2300
lstm_output_size = 64
embedding_size = 128
# Convolution
kernel_size = 5
filters = 64
pool_size = 3

model1 = Sequential()
model1.add(Embedding(max_features, embedding_size, input_length=timesteps))
model1.add(Dropout(0.25))
model1.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model1.add(MaxPooling1D(pool_size=pool_size))
model1.add(LSTM(lstm_output_size))

model2 = Sequential()
model2.add(Embedding(max_features, embedding_size, input_length=timesteps))
model2.add(Dropout(0.25))
model2.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model2.add(MaxPooling1D(pool_size=pool_size))
model2.add(LSTM(lstm_output_size))

model=Sequential()
model.add(Merge([model1,model2],mode="cos")) #should output 2 values
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([train_query, train_doc], train_label, epochs=200,validation_data=([valid_query, valid_doc], valid_label))
pre=model.predict([test_query, test_doc])
print(pre)
