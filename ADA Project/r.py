# Popularity Based Recommendation System
# Content Based Recommendation System
# Collaborative Based Recommendation System
import pandas as pd
import numpy as np
import difflib
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Remove SettingWithCopyWarning
pd.options.mode.chained_assignment = None

df = pd.read_csv("sdmn_video_details.csv")
# print(df.head())
# print(df.columns)
# c = list(df.columns)
# print(df['MAL_ID'].dtype)
# print(df['Name'].dtype)
# print(df['Score'].dtype)
# print(df['Genres'].dtype)
# print(df['English name'].dtype)
# print(df['Rating'])
# print(df['Ranked'])
# print(df['Popularity'])

# for i in range(len(c)):
#     if df[c[i]].dtype == 'object':
#         print(df[c[i]].unique())
# Rating,Ranked,Popularity,Members,Favorites,Watching,Completed,On-Hold,Dropped,Plan to Watch
# R - 17+ (violence & profanity),28.0,39,1251960,61971,105808,718161,71513,26678,329800

# # Content Based Recommendation System
# # select features
# features = ["Genres","English name"]
# df_selected = df
# # missing values
# # for f in features:
# #     index_names = df_selected[df_selected[f] == 'Unknown'].index
# #     df_selected.drop(index_names, inplace=True)
# #     df_selected[f].fillna('', inplace=True)
#
# # combining all 5 columns to 1
# combined_features = df_selected["English name"]+" "+df_selected["Genres"]
#
# # combined_features = pd.concat([df_selected["genres"], df_selected["keywords"], df_selected["tagline"],
# # df_selected["cast"], df_selected["director"]])
#
# # print(combined_features)
#
# # Regex
# def reg(content):
#     re_content = re.sub('[^a-zA-Z]',' ',content)
#     re_content = re_content.lower()
#     # re_content = re_content.split()
#     return re_content
#
# combined_features = combined_features.apply(reg)
# # print(combined_features)
#
# # Vectorize text
# vectorizer = TfidfVectorizer()
# f_vector = vectorizer.fit_transform(combined_features)
# # print(f_vector)
#
#
# # Cosine Similarity
# similarity = cosine_similarity(f_vector)
# # print(similarity)
# # print(similarity.shape)
#
# # Movie Recommendation System
# # Input
# print('\n\n\n')
# anime_name = input("Enter anime name:")
#
# # Creating list of all movie similar
# titles = df_selected["English name"].tolist()
# # print(titles)
#
# # finding close match
# find_close_match = difflib.get_close_matches(anime_name, titles)
# # print(find_close_match)
#
# close_match = find_close_match[0]
# # print(close_match)
#
# # finding index of movie with title
# i = df_selected[df_selected['English name'] == close_match]["anime_id"].values[0]
# # print(i)
#
# # getting list of similar movies
# similarity_score = list(enumerate(similarity[i]))
# # print(similarity_score)
#
# # sorting movies based on similarity score
# sorted_similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
# # print(sorted_similarity_score)
#
# # print movie names from above sim score
#
# print("Anime's suggested are:")
# i = 1
# for m in sorted_similarity_score:
#     index_m = m[0]
#     title = df_selected[df_selected.index == index_m]["English name"].values[0]
#     if i < 15:
#         if title != 'Unknown':
#             print(i, '.', title)
#             i += 1
#
# # print('\n\n\n')
# # f = [229170.0,182126.0,131625.0,62330.0,20688.0,8904.0,3184.0,1357.0,741.0,1580.0]
# # g = list(reversed(range(1,11)))
# # print(g)
# #
# # s = []
# # for o in range(0,10):
# #      s.append(f[o]*g[o])
# #
# # print(s)
# # su = sum(s)
# # print(su)
# # print(su/55)
# # print(su/sum(f))
# # print(su/(55*sum(f)))
# # p = (sum(f))
# # i = 10
# # s = 0
# # for u in f:
# #     s = s+(u*i)
# #     i = i-1
# #
# # h = s/(p*55)
# # print(h)

# # Popularity Based Recommendation System
# features = ['English name', 'Japanese name', "Score", "Members", "Score-10", "Score-9", "Score-8", "Score-7", "Score-6",
#             "Score-5", "Score-4", "Score-3", "Score-2", "Score-1"]
# df_selected = df[features]
# # print(df_selected.head())
# # print(df_selected.shape)
# # missing values
# for f in features:
#     for i in range(len(features)):
#         # Delete these row indexes from dataFrame
#         index_names = df_selected[df_selected[features[i]] == 'Unknown'].index
#         df_selected.drop(index_names, inplace=True)
#         # df_selected = df_selected.drop([df[]])
#         df_selected[f].fillna('', inplace=True)
#         # df_selected.replace({f:{'Unknown':''}},inplace=True)
#
# # print(df_selected.shape)
# rating_columns = list(df_selected.iloc[:, 4:].columns)
# # print(rating_columns)
#
#


#
# def weighted_rating(dt):
#     s = 0
#     for j in reversed(range(1, 11)):
#         v = dt[rating_columns[10 - j]].apply(lambda x: float(x)) * j
#         s = s + v
#         # print(s)
#     return s / dt['Members'].apply(lambda x: float(x))


# weighted_rating(df_selected)
#
# df_selected['Weighted_Score'] = weighted_rating(df_selected)
# # print(df_selected['weighted_score'])
# # print(df_selected['Score'])
#
# # Sorting movies based on score calculated above
# # df_selected = df_selected.sort_values('score', ascending=False)
# df_selected = df_selected.sort_values('Score', ascending=False)
#
# # # Printing the top 15 movies
# print('\n\n\n')
# print("The top 15 movies based on ratings:")
# print(df_selected[['English name', 'Japanese name', 'Score','Weighted_Score']].reset_index(drop=True).head(15))
#
# # p1 = [312.0,529.0,1242.0,1713.0,1068.0,634.0,265.0,83.0,50.0,27.0] # 6.98
# # p2 = [229170.0,182126.0,131625.0,62330.0,20688.0,8904.0,3184.0,1357.0,741.0,1580.0] # 8.78
# # o = [10,9,8,7,6,5,4,3,2,1]
# # d1 = []
# # d2 = []
# # for i in o:
# #       d1.append(p1[10-i]*i)
# #       d2.append(p2[10-i]*i)
# #
# # print(d1)
# # print(sum(d1))
# # print(sum(d1)/13224)
# # print(d2)
# # print(sum(d2))#5608661.0
# # print(sum(d2)/1251960)
# # s1 = sum(p1)
# # s2 = sum(p2)
# # print(s1)
# # print(s1/13224)
# # print(s2)
# # print(s2/1251960)
# # 3710.0,4369,13224,18,642,7314,766,1108,3394,312.0,529.0,1242.0,1713.0,1068.0,634.0,265.0,83.0,50.0,27.0
# # 28.0,39,1251960,61971,105808,718161,71513,26678,329800,229170.0,182126.0,131625.0,62330.0,20688.0,8904.0,3184.0,1357.0,741.0,1580.0


# Collaborative Based Recommendation System
rating_data = pd.read_csv("D:\\Users\\Me\\Programming\\My_Python\\NLP\\my_p's\\datasets\\animelist.csv\\animelist.csv", nrows=50000000)
anima_data = pd.read_csv("D:\\Users\\Me\\Programming\\My_Python\\NLP\\my_p's\\datasets\\anime.csv\\anime.csv")
print(anima_data.shape)
print(rating_data.shape)
# To save my Pc time I made a new data where i just fetch anime_id(MAL_ID) and Name so that i can use merge() function on it.
# anima_data = anima_data.rename(columns={"MAL_ID": "anime_id"})
anima_contact_data = anima_data[["anime_id", "Name"]]

rating_data = rating_data.merge(anima_contact_data, left_on='anime_id', right_on='anime_id', how='left')
rating_data = rating_data[["user_id", "Name", "anime_id", "rating", "watching_status", "watched_episodes"]]
# print(rating_data.head())

count = rating_data['user_id'].value_counts()
count1 = rating_data['anime_id'].value_counts()
# print(count[count >= 500].index)
# print(count[count >= 500])
# print(count1[count1 >= 200].index)
# print(count)
# print(count1)
rating_data = rating_data[rating_data['user_id'].isin(count[count >= 500].index)].copy()
# print(rating_data.tail())
rating_data = rating_data[rating_data['anime_id'].isin(count1[count1 >= 200].index)].copy()
# print(rating_data.tail())
# print(rating_data.shape)

combine_movie_rating = rating_data.dropna(axis=0, subset=['Name'])
movie_ratingCount = (combine_movie_rating.groupby(by=['Name'])['rating'].count().reset_index()[['Name', 'rating']])
# print(combine_movie_rating.head())
# print(movie_ratingCount.head())

rating_data = combine_movie_rating.merge(movie_ratingCount, left_on='Name', right_on='Name', how='left')
rating_data = rating_data.drop(columns="rating_x")
rating_data = rating_data.rename(columns={"rating_y": "rating"})
# print(rating_data.head())

# Encoding categorical data
user_ids = rating_data["user_id"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded2user = {i: x for i, x in enumerate(user_ids)}
rating_data["user"] = rating_data["user_id"].map(user2user_encoded)
n_users = len(user2user_encoded)

anime_ids = rating_data["anime_id"].unique().tolist()
anime2anime_encoded = {x: i for i, x in enumerate(anime_ids)}
anime_encoded2anime = {i: x for i, x in enumerate(anime_ids)}
rating_data["anime"] = rating_data["anime_id"].map(anime2anime_encoded)
n_animes = len(anime2anime_encoded)

# print(f"Num of users: {n_users}, Num of animes: {n_animes}")
# print("Min total rating: {}, Max total rating: {}".format(min(rating_data['rating']), max(rating_data['rating'])))

g = rating_data.groupby('user_id')['rating'].count()
top_users = g.dropna().sort_values(ascending=False)[:20]
top_r = rating_data.join(top_users, rsuffix='_r', how='inner', on='user_id')

g = rating_data.groupby('anime_id')['rating'].count()
top_animes = g.dropna().sort_values(ascending=False)[:20]
top_r = top_r.join(top_animes, rsuffix='_r', how='inner', on='anime_id')

pivot = pd.crosstab(top_r.user_id, top_r.anime_id, top_r.rating, aggfunc=np.sum)
pivot.fillna(0, inplace=True)

piviot_table = rating_data.pivot_table(index="Name", columns="user_id", values="rating").fillna(0)
# print(piviot_table)

from scipy.sparse import csr_matrix

piviot_table_matrix = csr_matrix(piviot_table.values)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

model = NearestNeighbors(metric="cosine", algorithm="brute")
model.fit(piviot_table_matrix)
NearestNeighbors(algorithm='brute', metric='cosine')

print('\n\n\n')
def predict():
    random_anime = np.random.choice(piviot_table.shape[0])
    # print(random_anime)
    # This will choose a random anime name and our model will predict on it.

    query = piviot_table.iloc[random_anime, :].values.reshape(1, -1)
    distance, suggestions = model.kneighbors(query, n_neighbors=6)

    for o in range(0, len(distance.flatten())):
        if o == 0:
            print('Recommendations for {0}:\n'.format(piviot_table.index[random_anime]))
        else:
            print('{0}: {1}, with distance of {2}:'.format(o, piviot_table.index[suggestions.flatten()[o]],
                                                           distance.flatten()[o]))

predict()
