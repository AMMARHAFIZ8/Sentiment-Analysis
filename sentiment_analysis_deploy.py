# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:03:48 2022

@author: ACER
"""
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pickle
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# 1) trained model --> loading from h5
# 2) tokenizer
# 3) MMS?OHE --> loading pickle

#%% Ammar
# Deployment ususally done on another PC/mobile

# to load trained model 
loaded_model = load_model(os.path.join(os.getcwd(),"model.h5"))

loaded_model.summary()

#to load tokenozer
TOKENIZER_PATH = os.path.join(os.getcwd(), 'tokenizer_sentiment.json')
with open (TOKENIZER_PATH,'r') as json_file:
    loaded_tokenizer = json.load(json_file)


OHE_PATH = os.path.join(os.getcwd(), 'ohe.pkl')
# #to load ohe
with open(OHE_PATH,'rb')as file:
    loaded_ohe = pickle.load(file)



#%% Warren

#to load tokenozer
TOKENIZER_PATH = os.path.join(os.getcwd(),"warren's_model" ,'tokenizer_sentiment.json')
with open (TOKENIZER_PATH,'r') as json_file:
    loaded_tokenizer_warren = json.load(json_file)


OHE_PATH = os.path.join(os.getcwd(),"warren's_model" , 'ohe.pkl')
# #to load ohe
with open(OHE_PATH,'rb')as file:
    loaded_ohe_warren = pickle.load(file)

# to load trained model 
loaded_model_warren = load_model(os.path.join(os.getcwd(),"warren's_model" ,"model.h5"))

loaded_model_warren.summary()

#%% Divyah

#to load tokenozer
TOKENIZER_PATH = os.path.join(os.getcwd(),"Divyah's_model" ,'tokenizer_sentiment.json')
with open (TOKENIZER_PATH,'r') as json_file:
    loaded_tokenizer_Divyah = json.load(json_file)


OHE_PATH = os.path.join(os.getcwd(),"Divyah's_model" , 'ohe.pkl')
# #to load ohe
with open(OHE_PATH,'rb')as file:
    loaded_ohe_Divyah = pickle.load(file)

# to load trained model 
loaded_model_Divyah = load_model(os.path.join(os.getcwd(),"Divyah's_model" ,"model.h5"))

loaded_model_Divyah.summary()


# Preprocessing

while True: # Forever loop #beware may overload
    input_review = input(" Review : ")
    
    input_review = re.sub('<.*?>',' ',input_review)
    input_review = re.sub('[^a-zA-Z]',' ',input_review).lower().split()
    
    #ammar
    tokenizer = tokenizer_from_json(loaded_tokenizer)
    input_review_encoded = tokenizer.texts_to_sequences(input_review)
    
    input_review_encoded = pad_sequences(np.array(input_review_encoded).T,
                                         maxlen=180,
                                         padding='post',truncating='post')
    
    outcome = loaded_model.predict(np.expand_dims(input_review_encoded,axis=-1))
    
    
    print("Ammar's model says the review is " + loaded_ohe.inverse_transform(outcome))
    
    
    #divyah
    tokenizer = tokenizer_from_json(loaded_tokenizer_Divyah)
    input_review_encoded = tokenizer.texts_to_sequences(input_review)
    
    input_review_encoded = pad_sequences(np.array(input_review_encoded).T,
                                          maxlen=180,
                                          padding='post',truncating='post')
    
    outcome = loaded_model_Divyah.predict(np.expand_dims(input_review_encoded,axis=-1))
    
    
    print("Divyah's model says the review is " + loaded_ohe_Divyah.inverse_transform(outcome))
    
    
    #  Warren
    tokenizer = tokenizer_from_json(loaded_tokenizer_warren)
    input_review_encoded = tokenizer.texts_to_sequences(input_review)
    
    input_review_encoded = pad_sequences(np.array(input_review_encoded).T,
                                          maxlen=180,
                                          padding='post',truncating='post')
    
    outcome = loaded_model_warren.predict(np.expand_dims(input_review_encoded,axis=-1))
    
    
    print("Warren's model says the review is " + loaded_ohe_warren.inverse_transform(outcome))




