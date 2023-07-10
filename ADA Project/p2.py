import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from keras import layers
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split as tt
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Twitter Climate Change Sentiment Dataset

# df = pd.read_csv("D:\\Users\\Me\\Programming\\My_Python\\NLP\\my_p's\\datasets\\twitter_sentiment\\twitter_sentiment_data.csv")

df = pd.read_csv("D:\\Users\\Me\\Programming\\My_Python\\NLP\\my_p's\\datasets\\sentiment_analysis\\Twitter_Data.csv")

# #  0 Indicating it is a Neutral Tweet/Comment
# #  1 Indicating a Positive Sentiment
# # -1 Indicating a Negative Tweet/Comment
# #  2 News
# labels = list(df['sentiment'].unique())
#
# # Removing bias
# neutral = df[df['sentiment'] == 0]
# pos = df[df['sentiment'] == 1]
# neg = df[df['sentiment'] == -1]
# news = df[df['sentiment'] == 2]
#
# dt = pd.concat([neutral.iloc[:3990, ], pos.iloc[:3990, ], neg.iloc[:3990, ], news.iloc[:3990, ]])
# # dt = df

# Stemming
words = stopwords.words("english")
# port_stem = PorterStemmer()
#
#
# # def stemming(contents):
# #     stemmed_contents = re.sub(r'[^a-zA-Z]', ' ', contents)
# #     stemmed_contents = stemmed_contents.lower()
# #     stemmed_contents = stemmed_contents.split()
# #     stemmed_contents = [port_stem.stem(word) for word in stemmed_contents if word not in words]
# #     stemmed_contents = ' '.join(stemmed_contents)
# #     return stemmed_contents
# #
# #
# # dt["message"] = dt["message"].apply(stemming)
# # print(dt['message'])
# #  0 Indicating it is a Neutral Tweet/Comment
# #  1 Indicating a Positive Sentiment -> 2
# # -1 Indicating a Negative Tweet/Comment -> 1
# #  2 News -> 3
# ma = {-1:1,0:0,1:2,2:3}
# dt.replace({'sentiment':{-1:1,0:0,1:2,2:3}},inplace=True)
# n = dt['sentiment']

# # Word Cloud
# text = ''
# for news in dt['message']:
#     text += f" {news}"
#
# wordcloud = WordCloud(width=3000,height=2000,background_color='black',stopwords=set(nltk.corpus.stopwords.words("english"))).generate(text)
# fig = plt.figure(figsize=(40, 30),facecolor='k',edgecolor='k')
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.tight_layout(pad=0)
# plt.show()
# del text
n = df['sentiment']
df["message"] = df["message"].astype(str)
x = []
# stop_words = set(nltk.corpus.stopwords.words("english"))
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
for par in df["message"].values:
    tmp = []
    sentences = nltk.sent_tokenize(par)
    for sent in sentences:
        sent = sent.lower()
        tokens = tokenizer.tokenize(sent)
        filtered_words = [w.strip() for w in tokens if w not in words and len(w) > 1]
        tmp.extend(filtered_words)
    x.append(tmp)


import gensim

EMBEDDING_DIM = 100
w2v_model = gensim.models.Word2Vec(sentences=x, vector_size=EMBEDDING_DIM, window=5, min_count=1)
# rock_idx = w2v_model.wv.key_to_index["rock"]
# rock_cnt = w2v_model.wv.get_vecattr("rock", "count")
# print(rock_idx)
# print(rock_cnt)
vocab_len = len(w2v_model.wv)
print(vocab_len)
# for u in w2v_model.wv:
#     print(u)


# print(w2v_model.wv["donald"])
# print(w2v_model.wv.most_similar("donald"))

# Tokenizing Text -> Representing each word by a number
# Mapping of original word to number is preserved in word_index property of tokenizer
# Tokenized applies basic processing like changing it to lower case, explicitly setting that as False
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)
X = tokenizer.texts_to_sequences(x)

# let's check the first 10 words of first news every word has been represented with a number
# Let's check few word to numerical representation
# Mapping is preserved in dictionary -> word_index property of instance

word_index = tokenizer.word_index
for word, num in word_index.items():
    print(f"{word} -> {num}")
    if num == 10:
        break


# Making histogram for no of words in news shows that most news article are under 700 words.
# Let's keep each news small and truncate all news to 700 while tokenizing
# plt.hist([len(x) for x in X], bins=500)
# plt.show()
#
# It's heavily skewed. There are news with 5000 words? Let's truncate these outliers :)
nos = np.array([len(x) for x in X])
# print(max(nos)) = 26
# print(min(nos)) = 1

# print(len(nos)) = 15960
# print(len(nos[nos < 16])) = 12684
# print(len(nos[nos < 17])) = 13782
# # Out of 48k news, 44k have less than 700 words
#
# # Let's keep all news to 700, add padding to news with less than 700 words and truncating long ones
maxlen = 16
# print(X)

# Making all news of size maxlen defined above
X = pad_sequences(X, maxlen=maxlen)
# print(X.shape)

# all news has 700 words (in numerical form now). If they had less words, they have been padded with 0
# 0 is not associated to any word, as mapping of words started from 1
# 0 will also be used later, if unknowns word is encountered in test set
# print(len(X[0])) = 17

# Adding 1 because of reserved 0 index
# Embedding Layer creates one more vector for "UNKNOWN" words, or padded words (0s). This Vector is filled with zeros.
# Thus, our vocab size increases by 1
v_s = len(tokenizer.word_index) + 1


# Function to create weight matrix from word2vec gensim model
def get_weight_matrix(m, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    # step vocab, store vectors using the Tokenizers integer mapping
    for w, i in vocab.items():
        weight_matrix[i] = m.wv[w]
    return weight_matrix


# Getting embedding vectors from word2vec and using it as weights of non-trainable keras embedding layer
embedding_vectors = get_weight_matrix(w2v_model, word_index)
# Defining Neural Network
# model = Sequential()
# # Non-trainable embedding layer
# model.add(Embedding(v_s, output_dim=EMBEDDING_DIM, weights=[embedding_vectors], input_length=maxlen, trainable=False))


# # # LSTM
# model = Sequential()
# model.add(Embedding(v_s, EMBEDDING_DIM, input_length=X.shape[1]))
# # model.add(keras.SpatialDropout1D(0.2))
# model.add(LSTM(100))
# model.add(Dense(4, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#
# m_train,m_test,n_train,n_test = tt(X, n)
# history = model.fit(m_train, n_train, epochs=5, batch_size=32)

# model.add(LSTM(units=128))
# model.add(Dense(4,activation='relu'))
# model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
#
# del embedding_vectors
# model.summary()
#
# # Train test split
# m_train,m_test,n_train,n_test = tt(X, n,stratify=n,random_state=0)
# model.fit(m_train, n_train, batch_size=32,epochs=5, verbose=2)

# # Prediction is in probability of news being real, so converting into classes
# # Class 0 (Fake) if predicted prob < 0.5, else class 1 (Real)
# y_pred = (model.predict(X_test)).astype("int")
# accuracy_score(y_test, y_pred)
# print(classification_report(y_test, y_pred))
#
# del model

# Functional API (a bit more flexible)
model = keras.Sequential()
model.add(Embedding(v_s, output_dim=EMBEDDING_DIM, weights=[embedding_vectors], input_length=maxlen, trainable=False))
model.add(keras.Input(shape=maxlen))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(4))
ip = keras.Input(shape=maxlen)
x = layers.Dense(512,activation='relu',name='First_layer')(ip)
x = layers.Dense(256,activation='relu',name='Second_layer')(x)
y = layers.Dense(4,activation='softmax')(x)
model = keras.Model(inputs=ip,outputs=y)

model.compile(
    # Sequential API (output layer no activation)
    # loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # Functional API
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy'],
)


m_train,m_test,n_train,n_test = tt(X, n,stratify=n,random_state=0)
model.fit(m_train,n_train, batch_size=32,epochs=5, verbose=2)
model.evaluate(m_test,n_test, batch_size=32, verbose=2)
