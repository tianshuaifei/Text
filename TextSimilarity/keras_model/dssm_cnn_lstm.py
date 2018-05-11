

from keras.layers import Conv1D, MaxPooling1D
import keras
from keras.layers import Input, LSTM, Dense,Embedding
from keras.models import Model


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

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

tweet_a = Input(shape=(timesteps,))
tweet_b = Input(shape=(timesteps,))

shared_emb=Embedding(output_dim=512, input_dim=max_features, input_length=20)
shared_cnn = Conv1D(filters,kernel_size,padding='valid', activation='relu',strides=1)
shared_p=MaxPooling1D(pool_size=pool_size)
shared_lstm = LSTM(lstm_output_size)


tweet_a_emb = shared_emb(tweet_a)
tweet_b_emb= shared_emb(tweet_b)

encoded_a = shared_cnn(tweet_a_emb)
encoded_b = shared_cnn(tweet_b_emb)

encoded_a_p=shared_p(encoded_a)
encoded_b_p=shared_p(encoded_b)

encoded_a_lstm = shared_lstm(encoded_a_p)
encoded_b_lstm = shared_lstm(encoded_b_p)


merged_vector = keras.layers.dot([encoded_a_lstm, encoded_b_lstm],axes=-1)
# And add a logistic regression on top
predictions = Dense(num_classes, activation='softmax')(merged_vector)


model = Model(inputs=[encoded_a_lstm, encoded_b_lstm], outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([train_query, train_doc], train_label, epochs=200,validation_data=([valid_query, valid_doc], valid_label))
pre=model.predict([test_query, test_doc])
print(pre)
model.save('dssm_cnn_lstm_model.h5')
