# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:54:29 2020

@author: Ines
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def read_file(filepath):
    file = open(os.path.join(filepath), mode='rt', encoding='utf-8')
    content = file.read()
    file.close()
    lines = content.strip().split('\n')
    lines = [i.split('\t') for i in lines]
    eng_deu_lines = np.array(lines)
   
    eng_deu_lines = eng_deu_lines[:50000, :]
    eng_deu_lines = eng_deu_lines[:, 0:2]
    df = pd.DataFrame({'eng': eng_deu_lines[:, 0], 'deu': eng_deu_lines[:, 1]})
    return df, eng_deu_lines


def process_data(df):
    dfp = df
    # Put everything to lower case
    dfp['eng'] = dfp["eng"].str.lower()
    dfp['deu'] = dfp["deu"].str.lower()
    dfp['eng'] = dfp['eng'].str.replace('[^\w\s]', '')
    dfp['deu'] = dfp['deu'].str.replace('[^\w\s]', '')
    return dfp


def data_array(df):
    eng = df['eng'].tolist()
    deu = df['deu'].tolist()
    eng_deu = []
    for i in range(len(eng)):
        eng_deu.append([eng[i], deu[i]])
    return np.array(eng_deu)


def visualise_data(df):
    # See the  length of words of our dataset
    # For that we slip the string and append the length of the word in a list   
    len_eng_word = []
    len_deu_word = []
    for i in df['eng']:
        len_eng_word.append(len(i.split()))
    for i in df['deu']:
        len_deu_word.append(len(i.split())) 
    len_deu_word_df = pd.DataFrame({'len_eng': len_eng_word, 'len_deu': len_deu_word})
    plt.subplot(3, 1, 1)
    len_deu_word_df['len_eng'].hist(bins=30)
    plt.title("Distribution of english length words")
    plt.xlabel('length of word')
    plt.ylabel('number of words')
    plt.subplot(3, 1, 3)
    len_deu_word_df['len_deu'].hist(bins=30)
    plt.title("Distribution of deutsh length words")
    plt.xlabel('length of word')
    plt.ylabel('number of words')
    plt.show()


def token(content):
    tok = Tokenizer()
    tok.fit_on_texts(content)
    return tok


def encoding(content, len, tok):
    sequences = tok.texts_to_sequences(content)
    sequences = pad_sequences(sequences, maxlen=len, padding='post')
    return sequences


def get_word(n, tok):
    for word, idx in tok.word_index.items():
        if idx == n:
            return word
    return None


def prediction(predictions, eng_tok):
    preds = []
    for i in predictions:
        tmp = []
        for j in range(len(i)):
            w = get_word(i[j], eng_tok)
            if j>0:
                if(w == get_word(i[j-1],eng_tok)) or if(w == None):
                    tmp.append("")
                else:
                    tmp.append(w)
            else:
                if w is None:
                    tmp.append("")
                else:
                    tmp.append(t)
        preds.append("".join(tmp))
    df_preds = pd.DataFrame({'Predicted': preds, 'Actual': data_test[:, 0]})
    return df_preds




