from __future__ import absolute_import
from __future__ import print_function

import keras
from keras.layers import Input, LSTM, Dense,Embedding
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

tweet_a = Input(shape=(timesteps,))
tweet_b = Input(shape=(timesteps,))


tweet_a_emb = Embedding(output_dim=512, input_dim=max_features, input_length=20)(tweet_a)
tweet_b_emb= Embedding(output_dim=512, input_dim=max_features, input_length=20)(tweet_b)
# This layer can take as input a matrix
# and will return a vector of size 64
shared_lstm = LSTM(64)

# When we reuse the same layer instance
# multiple times, the weights of the layer
# are also being reused
# (it is effectively *the same* layer)
encoded_a = shared_lstm(tweet_a_emb)
encoded_b = shared_lstm(tweet_b_emb)

# We can then concatenate the two vectors:
# merged_vector =Cosine([encoded_a, encoded_b])
# merged_vector_em=Reshape(2)
#merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)
merged_vector = keras.layers.dot([encoded_a, encoded_b],axes=-1)
# And add a logistic regression on top
predictions = Dense(num_classes, activation='softmax')(merged_vector)

# We define a trainable model linking the
# tweet inputs to the predictions
model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([train_query, train_doc], train_label, epochs=10,validation_data=([valid_query, valid_doc], valid_label))
pre=model.predict([test_query, test_doc])
print(pre)
