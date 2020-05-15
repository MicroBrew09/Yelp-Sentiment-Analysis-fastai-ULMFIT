

#Adapted from https://www.analyticsvidhya.com/blog/2018/11/tutorial-text-classification-ulmfit-fastai-library/

import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import nltk
from nltk.corpus import stopwords
import fastai
from fastai import *
from fastai.text import *
import pandas as pd
import numpy as np
from functools import partial
from IPython.display import display, HTML
import io
import os
import json
from langdetect import detect
from langdetect import DetectorFactory, detect_langs
import sys, os, cv2
from sklearn.model_selection import train_test_split
#from libsixel.encoder import Encoder, SIXEL_OPTFLAG_WIDTH, SIXEL_OPTFLAG_COLORS

def main(input1,input2):


 for file in os.listdir(input2):
 	df = pd.read_csv(input1)
 	print(file)
 	DetectorFactory.seed = 0 
 	rev = pd.read_json(input2+file, orient='record', typ='frame', dtype=dict, precise_float=False, lines=True)
 	rev = pd.DataFrame(rev)
 	df = pd.DataFrame({'label':df.label, 'text':df.text})
 	rev['text'] = rev['text'].str.replace("[^a-zA-Z]", " ")
 	df['text'] = df['text'].str.replace("[^a-zA-Z]", " ")
 	rev['text'] = rev.text.astype(str).str.lower()
 	df.text = df.text.astype(str).str.lower()
 	tokenized_doc1 = rev['text'].apply(lambda x: x.split())
 	tokenized_doc = df['text'].apply(lambda x: x.split())
 	stop_words = stopwords.words('english')
 	tokenized_doc1 = tokenized_doc1.apply(lambda x: [item for item in x if item not in stop_words])
 	tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

 	detokenized_doc = []
 	for i in range(len(df)):
 		t = ' '.join(tokenized_doc[i])
 		detokenized_doc.append(t)

 	detokenized_doc1 = []
 	for i in range(len(rev)):
 		t = ' '.join(tokenized_doc1[i])
 		detokenized_doc1.append(t)

 	df['text'] = detokenized_doc
 	rev['text'] = detokenized_doc1
 	df_trn, df_val = train_test_split(df, stratify = df['label'], test_size = 0.2, random_state = 2)
 	data_lm = TextLMDataBunch.from_df(train_df = df_trn, valid_df = df_val, path = "")
 	data_classification = TextClasDataBunch.from_df(path = "", train_df = df_trn, valid_df = df_val, vocab=data_lm.train_ds.vocab, bs=32)
 	data_classification = TextClasDataBunch.from_df(path = "", train_df = df_trn, valid_df = df_val, vocab=data_lm.train_ds.vocab, bs=32)

 	learn = language_model_learner(data=data_lm, arch=AWD_LSTM, drop_mult=0.2)
 	learn = text_classifier_learner(data_classification, arch=AWD_LSTM, drop_mult=0.2)
 	learn.load('p_class')
 	lrn = learn.load('p_class')
 	learn = learn.load_encoder('ft_enc_p')
 	preds,y,losses = learn.get_preds(with_loss=True)
 	N = len(rev)

 	reviews = rev['text'].apply(lambda x: x.split())
 	price_rating = []
 	prob1 = []
 	prob2 = []
 	prob3 = []
 	j=0
 	for j in range(len(rev['text'])):
 		try:
 			language = detect(rev['text'][j])		
 			if detect(rev['text'][j]) == "en":
	 			pred, T, prob = lrn.predict(reviews[j])
 				price_rating.append(pred)
 				prob1.append(prob[0]*100)
 				prob2.append(prob[1]*100)
 				prob3.append(prob[2]*100)
 				print(j,pred, prob[0]*100, prob[1]*100, prob[2]*100)
 			else:
 				price_rating.append(2)
 				prob1.append(0)
 				prob2.append(0)
 				prob3.append(0)
 				print(j,pred, 0, 0, 0)
 		except:
 			language = "error"
 			price_rating.append(2)
 			prob1.append(0)
 			prob2.append(0)
 			prob3.append(0)
 			print(j,pred, 0, 0, 0)

 	rev['price'] = price_rating
 	rev['p_prob1'] = prob1
 	rev['p_prob2'] = prob2
 	rev['p_prob3'] = prob3
 	rev.to_csv(r''+input2+"p_"+file)
 	print(rev)
 	print("Completed"+file)

if __name__ == '__main__':
    input1 = sys.argv[1] #Either service or price
    input2 = sys.argv[2]
    main(input1,input2)



