import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,PorterStemmer
import re
from nltk.corpus import words
# nltk.download()
sw = stopwords.words('english')

if 'us' in sw:
    print('Fuck')

port_stem = PorterStemmer()
lem = WordNetLemmatizer()
print(port_stem.stem('us'))
print(lem.lemmatize('us'))


# def stemming(contents):
#     stemmed_contents = re.sub(r'[^a-zA-Z]',' ',contents)
#     stemmed_contents = stemmed_contents.lower()
#     stemmed_contents = stemmed_contents.split()
#     stemmed_contents = [lem.lemmatize(word) for word in stemmed_contents if word not in sw]
#     stemmed_contents = ' '.join(stemmed_contents)
#     return stemmed_contents
#
# df["review"] = df["review"].apply(stemming)

# from sklearn.model_selection import train_test_split as tt
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score
# import warnings
# warnings.filterwarnings('ignore')
#
# df = pd.read_csv('channel_metrics.csv')
#
# x = np.asarray(df.durationSecs).reshape(-1,1)
# y = np.asarray(df.views).reshape(-1,1)
#
# x_train,x_test,y_train,y_test = tt(x,y,train_size=0.8,random_state=0)
#
# log_reg = LogisticRegression()
# rf_reg = RandomForestRegressor()
# log_reg.fit(x_train,y_train)
# y_train_pred = log_reg.predict(x_train)
# y_test_pred = log_reg.predict(x_test)
#
# train_error = r2_score(y_train,y_train_pred)
# test_error = r2_score(y_train,y_train_pred)
# print('Error in train:',train_error)
# print('Error in test:',test_error)
#
# # rf_reg.fit(x_train,y_train)
# # y_train_pred = rf_reg.predict(x_train)
# # y_test_pred = rf_reg.predict(x_test)
# #
# # train_error = r2_score(y_train,y_train_pred)
# # test_error = r2_score(y_train,y_train_pred)
# # print('Error in train:',train_error)
# # print('Error in test:',test_error)