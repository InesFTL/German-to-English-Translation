# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:54:29 2020

@author: Ines
"""
from data_processing import read_file, process_data, data_array, token,\
    encoding, prediction
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model


def main(filepath, modelpath):

    """ main function 

    Args:
        filepath ([txt]): filepath containing our dataset
        modelpath ([h5]): modelpath containing our model
    """
    df, eng_deu_lines = read_file(filepath)
    dfp = process_data(df)
    eng_deu = data_array(dfp)
    eng_tok = token(eng_deu[:, 0])
    eng_len_vocab = len(eng_tok.word_index) + 1
    deu_tok = token((eng_deu[:, 1]))
    deu_len_vocab = len(deu_tok.word_index) + 1
    data_train, data_test = train_test_split(eng_deu, test_size=0.2,  
                                             random_state=1)
    X_train = encoding(data_train[:, 1], 8, deu_tok)
    y_train = encoding(data_train[:, 0], 8, eng_tok)
    X_test = encoding(data_test[:, 1], 8, deu_tok)
    y_test = encoding(data_test[:, 1], 8, deu_tok)
    model = load_model(modelpath)
    preds = model.predict_classes(X_test.reshape((X_test.shape[0], 
                                                 X_test.shape[1])))
    df_preds = prediction(preds, eng_tok)
    return df_preds


if __name__ == "__main__":

    filepath = "deu.txt"
    modelpath = "model2.h5"
    df_preds = main(filepath, modelpath)
    df_preds.sample(5)
