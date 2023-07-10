from sklearn.feature_extraction.text import TfidfVectorizer
from minisom import MiniSom
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import DBSCAN
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from keras import layers
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split as tt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("sdmn_video_details.csv")

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
# df = pd.concat([neutral.iloc[:3990, ], pos.iloc[:3990, ], neg.iloc[:3990, ], news.iloc[:3990, ]])
# # df = df

# Stemming
words = stopwords.words("english")
port_stem = PorterStemmer()
lem = WordNetLemmatizer()


def stemming(contents):
    stemmed_contents = re.sub(r'[^a-zA-Z]', ' ', contents)
    stemmed_contents = stemmed_contents.lower()
    stemmed_contents = stemmed_contents.split()
    # stemmed_contents = [port_stem.stem(word) for word in stemmed_contents if word not in words]
    stemmed_contents = [lem.lemmatize(word) for word in stemmed_contents if word not in words]
    stemmed_contents = ' '.join(stemmed_contents)
    return stemmed_contents


# df["video_title"] = df["title"].apply(stemming)
df["video_title"] = df["title"]
print(df['video_title'])

# # ma = {-1:1,0:0,1:2,2:3}
# # df.replace({'sentiment':{-1:1,0:0,1:2,2:3}},inplace=True)
# # n = df['sentiment']
#
# # # Load the textual data
# # data = [
# #     "This is the first document",
# #     "This is the second document",
# #     "And this is the third one",
# #     "Is this the first document?",
# # ]
# #
# data = list(df['video_title'])
# # Convert the textual data to a vector representation using TF-IDF
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(data)
#
# # Define the size of the SOM and train the model
# som = MiniSom(3, 3, X.shape[1], sigma=1.0, learning_rate=0.5)
# som.random_weights_init(X.toarray())
# som.train_random(X.toarray(), 100)
#
# # Assign each document to a cluster based on the closest neuron
# clusters = {}
# for i, x in enumerate(X):
#     winner = som.winner(x.toarray()[0])
#     if winner not in clusters:
#         clusters[winner] = [i]
#     else:
#         clusters[winner].append(i)
#
# # # Print the clusters
# # for cluster, docs in clusters.items():
# #     print(f"Cluster {cluster}:")
# #     for doc in docs:
# #         print(f"  {data[doc]}")
#
# print(len(clusters))
#
# # Cluster the data using DBSCAN
# dbscan = DBSCAN(eps=0.5, min_samples=4)
# dbscan.fit(X)
#
# # Assign each document to a cluster based on the clustering result
# clusters = {}
# for i, label in enumerate(dbscan.labels_):
#     if label not in clusters:
#         clusters[label] = [i]
#     else:
#         clusters[label].append(i)
#
# print(len(clusters))
# # # Print the clusters
# # for cluster, docs in clusters.items():
# #     print(f"Cluster {cluster}:")
# #     for doc in docs:
# #         print(f"  {data[doc]}")
#
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(data)
#
# # Cluster the data using LDA
# lda = LatentDirichletAllocation(n_components=5)
# lda.fit(X)
#
# # Assign each document to a cluster based on the topic with the highest probability
# clusters = {}
# for i, x in enumerate(X):
#     topic = lda.transform(x)[0].argmax()
#     if topic not in clusters:
#         clusters[topic] = [i]
#     else:
#         clusters[topic].append(i)
#
# print(len(clusters))
# # # Print the clusters
# # for cluster, docs in clusters.items():
# #     print(f"Cluster {cluster}:")
# #     for doc in docs:
# #         print(f"  {data[doc]}")
#
# import torch
# from transformers import BertTokenizer, BertModel
# from sklearn.cluster import KMeans
#
# # Load the BERT model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
#
# # Tokenize the input data
# inputs = tokenizer(data, padding=True, truncation=True, return_tensors="pt")
#
# # Pass the tokenized input through the BERT model to get embeddings
# with torch.no_grad():
#     outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
# embeddings = outputs[0][:, 0, :].numpy()
#
# # Cluster the embeddings using K-Means
# kmeans = KMeans(n_clusters=5)
# kmeans.fit(embeddings)
#
# # Assign each document to a cluster based on the K-Means clustering result
# clusters = {}
# for i, label in enumerate(kmeans.labels_):
#     if label not in clusters:
#         clusters[label] = [i]
#     else:
#         clusters[label].append(i)
#
# print(len(clusters))
# # Print the clusters
# # for cluster, docs in clusters.items():
# #     print(f"Cluster {cluster}:")
# #     for doc in docs:
# #         print(f"  {data[doc]}")
