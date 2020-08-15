#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:54:53 2020

@author: Ines
"""

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from keras.callbacks import ModelCheckpoint
from keras import optimizers


def model_init(input, output, in_timesteps, out_timesteps, units):
    model = Sequential()
    model.add(Embedding(input, units, input_length=in_timesteps,  
                        mask_zero=True))
    model.add(LSTM(units))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(units, return_sequences=True))
    model.add(Dense(output, activation='softmax'))
    return model

   
def training_model(X_train, y_train, maxlen, 
                   deu_len_vocab, eng_len_vocab, filename):  
    model = model_init(deu_len_vocab, eng_len_vocab, maxlen, maxlen, 512)
    rms = optimizers.RMSprop(lr=0.001)
    model.compile(optimizer=rms, loss='sparse_categorical_crossentropy')
    filename = filename
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, 
                                 save_best_only=True, mode='min')
    H = model.fit(X_train, y_train.reshape(y_train.shape[0],  
                                           y_train.shape[1], 1), 
                  epochs=30, batch_size=512, validation_split=0.2, 
                  callbacks=[checkpoint], verbose=1)

    return H, filename

