#%%
import numpy as np
import matplotlib.pyplot as plt 
import string
import re,collections
import joblib 
import pickle as pkl
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from helper_code import *
from tqdm import tqdm_notebook
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
plt.style.use('ggplot')
# %%
model = joblib.load('Data/Models/final_model.joblib')
vectorizer = joblib.load('Data/Vectorizers/final_model.joblib')
# %%
text = 'okrem iného ako durič na brlohárenie'
text = preprocess_function(text)
text = [split_into_subwords(text)]
text_vectorized = vectorizer.transform(text)
model.predict(text_vectorized)
# %%
def open_file(filename):
    with open(filename,'r') as f:
        data = f.readlines()
    return data
# %%
data_raw = dict()
data_raw['sk'] = open_file('Data/Sentences/train_sentences.sk')
data_raw['cs'] = open_file('Data/Sentences/train_sentences.cs')
data_raw['en'] = open_file('Data/Sentences/train_sentences.en')
# %%
def show_statistics(data):
    for language,sentences in data.items():
        number_of_sentences = 0
        number_of_words = 0
        number_of_unique_words = 0
        sample_extract = ''
        word_list = ' '.join(sentences).split()
        
        
        print(f'Language: {language}')
        print('-----------------------')
        print(f'Number of sentences\t:\t {number_of_sentences}')
        print(f'Number of words\t\t:\t {number_of_words}')
        print(f'Number of unique words\t:\t {number_of_unique_words}')
        print(f'Sample extract\t\t:\t {sample_extract}...\n')
# %%
show_statistics(data_raw)
do_law_of_zipf(data_raw)
# %%
def preprocess(text):
    preprocessed_text = text.lower().replace('-',' ')
    translation_table = str.maketrans('\n',' ',string.punctuation + string.digits)
    preprocessed_text = preprocessed_text.translate(translation_table)
    return preprocessed_text
# %%
data_preprocessed = {k: [preprocess(sentence) for sentence in v] for k,v in data_raw.items()}
show_statistics(data_preprocessed)
# %%
sentences_train, y_train = [],[]
for k,v in data_preprocessed.items():
    for sentence in v:
        sentences_train.append(sentence)
        y_train.append(k)
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(sentences_train)
X_train
# %%
naive_classifier = MultinomialNB()
naive_classifier.fit(X_train,y_train)
# %%
data_val = dict()
data_val['sk'] = open_file('Data/Sentences/val_sentences.sk')
data_val['cs'] = open_file('Data/Sentences/val_sentences.cs')
data_val['en'] = open_file('Data/Sentences/val_sentences.en')

data_val_preprocessed = {k: [preprocess(sentence) for sentence in v] for k, v in data_val.items()}
# %%
sentences_val,y_val = [],[]
for k,v in data_val_preprocessed.items():
    for sentence in v:
        sentences_val.append(sentence)
        y_val.append(k)
X_val = vectorizer.transform(sentences_val)
predictions = naive_classifier.predict(X_val)
plot_confusion_matrix(y_val,predictions,['sk','cs','en'])
f1_score(y_val,predictions,average='weighted')
# %%
naive_classifier = MultinomialNB(alpha=0.0001,fit_prior=False)
naive_classifier.fit(X_train,y_train)
predictions = naive_classifier.predict(X_val)
plot_confusion_matrix(y_val,predictions,['sk','cs','en'])
f1_score(y_val,predictions,average='weighted')
# %%
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word,freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair,v_in):
    v_out = {}
    bigram = re.escape(" ".join(pair))
    p = re.compile(r'(?<!\S)'+bigram+r'(?1\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair),word)
        v_out[w_out] = v_in[word]
    return v_out
# %%
def get_vocab(data):
    words = []
    for sentence in data:
        words.extend(sentence.split())
    vocab = defaultdict(int)
    for word in words:
        vocab[' '.join(word)]
    return vocab

vocab = get_vocab(sentences_train)
# %%
"""for i in range(100):
    pairs = get_stats(vocab)
    best = max(pairs,key=pairs.get)
    vocab = merge_vocab(best,vocab)"""
# %%
merges = defaultdict(int)
for k,v in vocab.items():
    for subword in k.split():
        if len(subword) >= 2:
            merges[subword] += v

merge_ordered = sorted(merges,key=merges.get,reverse=True)
# %%
pkl.dump(merge_ordered,open('Data/Auxiliary/merge_ordered.pkl','wb'))
# %%
def split_into_subwords(text):
    merges = pkl.load(open('Data/Auxiliary/merge_ordered.pkl','rb'))
    subwords = []
    for word in text.split():
        for subword in merges:
            subword_count = word.count(subword)
            if subword_count > 0:
                word = word.replace([subword]*subword_count)
    return " ".join(subwords)

split_into_subwords('this is ari here')
# %%
data_preprocessed_subwords = {k:[split_into_subwords(sentence) for sentence in v] for k,v in data_preprocessed.items()}
show_statistics(data_preprocessed_subwords)
# %%
data_train_subwords = []
for sentence in sentences_train:
    data_train_subwords.append(split_into_subwords(sentence))
data_val_subwords = []
for sentence in sentences_val:
    data_val_subwords.append(split_into_subwords(sentence))
#%%
vectorizer = CountVectorizer()
# %%
X_train = vectorizer.fit_transform(data_train_subwords)
X_val = vectorizer.transform(data_val_subwords)
# %%
naive_classifier = MultinomialNB(fit_prior=False)
naive_classifier.fit(X_train,y_train)

predictions= naive_classifier.predict(X_val)
plot_confusion_matrix(y_val,predictions,['sk','cs','en'])
f1_score(y_val,predictions,average='weighted')
# %%
