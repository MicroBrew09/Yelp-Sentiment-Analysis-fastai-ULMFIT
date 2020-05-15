

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
from sklearn.model_selection import train_test_split
#from libsixel.encoder import Encoder, SIXEL_OPTFLAG_WIDTH, SIXEL_OPTFLAG_COLORS

def main(inputs):

	df = pd.read_csv(inputs)
	df = pd.DataFrame({'label':df.label, 'text':df.text})

	df['text'] = df['text'].str.replace("[^a-zA-Z]", " ") #Removing any punctuation, etc.  
	df.text = df.text.astype(str).str.lower()  
	tokenized_doc = df['text'].apply(lambda x: x.split())
	# remove stop-words
	#nltk.download('stopwords')
	stop_words = stopwords.words('english')
	tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
    
	# detokenization
	detokenized_doc = []
	for i in range(len(df)):
	    t = ' '.join(tokenized_doc[i])
	    detokenized_doc.append(t)
    
	df['text'] = detokenized_doc
	df_trn, df_val = train_test_split(df, stratify = df['label'], test_size = 0.2, random_state = 2)
     
	data_lm = TextLMDataBunch.from_df(train_df = df_trn, valid_df = df_val, path = "")
	data_classification = TextClasDataBunch.from_df(path = "", train_df = df_trn, valid_df = df_val, vocab=data_lm.train_ds.vocab, bs=32)
	learn = language_model_learner(data=data_lm, arch=AWD_LSTM, drop_mult=0.2)
	learn.fit_one_cycle(5, 1e-2)
	learn.unfreeze()
	learn.recorder.plot()
	learn.save_encoder('ft_enc_s')
	learn = text_classifier_learner(data_classification, arch=AWD_LSTM, drop_mult=0.2)
	learn.show_results() 
	learn.load_encoder('ft_enc_s')
	learn.lr_find(start_lr=1e-8, end_lr=1e2)
	learn.fit_one_cycle(10, 1e-2)
	learn.freeze()
	learn.fit_one_cycle(cyc_len=1, max_lr=1e-3, moms=(0.8, 0.7))
	learn.freeze_to(-2)
	learn.fit_one_cycle(1, slice(1e-4,1e-2), moms=(0.8,0.7))
	learn.freeze_to(-3)
	learn.fit_one_cycle(1, slice(1e-5,5e-3), moms=(0.8,0.7))
	learn.unfreeze()
	learn.fit_one_cycle(4, slice(1e-5,1e-3), moms=(0.8,0.7))
	learn.save('s_class')
	learn.show_results() 
	lrn = learn.load('s_class')

	preds, targets = lrn.get_preds()	
	predictions = np.argmax(preds, axis = 1)
	print(pd.crosstab(predictions, targets))
	print(pd.crosstab(predictions, targets).apply(lambda r: r/r.sum(), axis=1))
    #print(predictions, targets)
if __name__ == '__main__':
    inputs = sys.argv[1] #Either service or price or food
    main(inputs)



