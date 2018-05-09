from __future__ import absolute_import
from __future__ import print_function

import keras
from keras.layers import Input,Dense,Embedding,Conv1D,MaxPooling1D,GlobalAveragePooling1D
from keras.models import Model

from TextSimilarity.dssm_model import data_provider

print("Loading data...")
train_query, train_doc, train_label = data_provider.load_train_dataset()
valid_query, valid_doc, valid_label = data_provider.load_valid_dataset()
test_query, test_doc, test_label = data_provider.load_pre_dataset()
print(len(train_query), 'train sequences')

timesteps = 20
num_classes = 2
max_features=2300

import numpy as np
x_train = np.random.random((1000, timesteps, 128))
y_train = np.random.random((1000, num_classes))
print("train data shape : ", x_train.shape)

tweet_a = Input(shape=(timesteps,),name="query")
tweet_b = Input(shape=(timesteps,),name="doc")



shared_emb= Embedding(output_dim=512, input_dim=max_features, input_length=20,name="emb")
shared_cnn = Conv1D(64, 3, activation='relu')
shared_p=MaxPooling1D(3)
shared_g=GlobalAveragePooling1D()

tweet_a_emb = shared_emb(tweet_a)
tweet_b_emb= shared_emb(tweet_b)

encoded_a = shared_cnn(tweet_a_emb)
encoded_b = shared_cnn(tweet_b_emb)

encoded_a_p=shared_p(encoded_a)
encoded_b_p=shared_p(encoded_b)

encoded_a_p_g=shared_g(encoded_a_p)
encoded_b_p_g=shared_g(encoded_b_p)

merged_vector = keras.layers.dot([encoded_a_p_g, encoded_b_p_g],axes=-1)
# And add a logistic regression on top
predictions = Dense(num_classes, activation='softmax')(merged_vector)

# We define a trainable model linking the
# tweet inputs to the predictions
model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([train_query, train_query], train_label, epochs=10)
pre=model.predict([test_query, test_doc])
print(pre)
