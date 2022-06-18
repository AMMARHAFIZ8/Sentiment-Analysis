# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:17:43 2022

@author: ACER
"""
#%%

PATH = 'https://raw.github.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'


#%%
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import Input
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import plot_model
import pickle

# EDA
# Step 1 DATA LOADING

df = pd.read_csv(PATH)
df_copy = df.copy()


# Step 2 Data Inspection
df.head(10)
df.info()
df.describe()

df['sentiment'].unique() # to get unique targets
df['review'][5]
df['sentiment'][5]

df.duplicated().sum() #418
df[df.duplicated()]

# <br /> tags have to be removed
# Numbers/duplicates can be filtered



# Step 3 Data Cleaning

df = df.drop_duplicates()

# remove html tags
# '<br /> dj9ejdwujdpi2wjdpwp <br />'.replace('<br />',' ')
# df['review'][1].replace('<br />',' ')
review = df['review'].values #features = x
sentiment = df['sentiment'].values #sentiment = y


for index,rev in enumerate(review):
    #remove html tags
    # remove ? dont be greedy
    # zero or more occurance
    # any character except new line (/n)
    
    review[index] = re.sub('<.*?>',' ',rev)
    
    #convert lower case
    # remove numbers
    #^ means NOT
    review[index] = re.sub('[^a-zA-Z]',' ',rev).lower().split()


# Step 4 Features Selection
# Nothing to select

# Step 5 Preprocessing:
#       1 Convert into lower case
#       2 Tokenization
vocab_size = 10000
oov_token = 'OOV'


tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_token)


tokenizer.fit_on_texts(review) # to learn all of the words
word_index = tokenizer.word_index
print(word_index)

train_sequences = tokenizer.texts_to_sequences(review) # converts into numbers

#       3 Padding & truncating
length_of_reviews = [len(i)for i in train_sequences] #list comprehension
np.median(length_of_reviews) # get max length  for padding
print(np.median(length_of_reviews))

max_len = 100


padded_review = pad_sequences(train_sequences, maxlen=max_len, padding='post',
                              truncating='post')


#       4 One hot encoding
#only for sentiment

ohe = OneHotEncoder(sparse=False)
sentiment = ohe.fit_transform(np.expand_dims(sentiment, axis=-1))

#       5 Train test split

X_train, X_test, y_train, y_test = train_test_split(padded_review, sentiment,
                                                    test_size=0.3,
                                                    random_state=420)

X_train = np.expand_dims(X_train,axis=-1) # 2 to 3 dimensions
X_test = np.expand_dims(X_test,axis=-1)

#%% Model development

from tensorflow.keras.layers import Bidirectional, Embedding

embedding_dim = 64


model = Sequential()
model.add(Input(shape=(180))) #np.shape(X_train)[1:]
model.add(Embedding(vocab_size, embedding_dim))
model.add(Bidirectional(LSTM(embedding_dim,return_sequences=(True))))
# model.add(LSTM(128,return_sequences=(True)))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2,'softmax'))
model.summary()

plot_model(model)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='acc')

hist = model.fit(X_train, y_train, batch_size=128, epochs=10,
                 validation_data=(X_test, y_test))



hist.history.keys()

plt.figure()
plt.plot(hist.history['loss'],'r--', label= 'Training loss')
plt.plot(hist.history['val_loss'],'r--', label= 'Validation loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history['acc'],'r--', label = 'Training acc')
plt.plot(hist.history['val_acc'], label = 'Validation acc')
plt.legend()
plt.show()

#%% Model Evaluation
y_true = y_test
y_pred = model.predict(X_test)

#%%
y_true = np.argmax(y_true,axis=1)
y_pred = np.argmax(y_pred,axis=1)


#%%

print(classification_report(y_true,y_pred))
print(accuracy_score(y_true, y_pred))
print(confusion_matrix(y_true,y_pred))

#%% model saving

MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5') #machine learning , deep learning use h5, minmax use pickle
model.save(MODEL_SAVE_PATH)

import json
token_json = tokenizer.to_json()

TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_sentiment.json')
with open(TOKENIZER_PATH,'w') as file:
    json.dump(token_json,file)

OHE_PATH = os.path.join(os.getcwd(),'ohe.pkl')
with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe, file)



#%% Discussion/Reporting






























