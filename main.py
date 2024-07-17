import json
from datetime import date
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from data_prep_functions import extract_movie_info
from data_prep_functions import categorize_companies
from data_prep_functions import standardize_type_name
from data_prep_functions import standardize_nation_name
from data_prep_functions import standardize_ratings
from data_prep_functions import one_hot_encode_column
'''---------------------------------------------------------'''
'''Display setting'''
# display all columns
pd.set_option('display.max_seq_items', None)
pd.set_option('display.max_columns', None)
'''---------------------------------------------------------'''
'''Import data'''
with open('data/commercial_foreign.json') as f:
    data_cf = json.load(f)
f.close()

with open('data/commercial_korean.json') as f:
    data_ck = json.load(f)
f.close()

with open('data/noncommercial_foreign.json') as f:
    data_nf = json.load(f)
f.close()

with open('data/noncommercial_korean.json') as f:
    data_nk = json.load(f)
f.close()

with open('data/movie_info_1.json') as f:
    movie_info_1 = json.load(f)
f.close()

with open('data/movie_info_2.json') as f:
    movie_info_2 = json.load(f)
f.close()

'''------------------------------------------------------'''
'''Manipulate data'''
# read all movie info into one file, 'movie_info'
movie_info = movie_info_1 + movie_info_2
print(len(movie_info))

del movie_info_1
del movie_info_2

# extract relavant information from movie_info
movie_data = pd.DataFrame([extract_movie_info(movie) for movie in movie_info])

#drop empty values
movie_data = movie_data[(movie_data != '').all(axis=1)]

# print stats for movie_data
print(f"Total movie count: {len(movie_data)}")
print(f"samples: {movie_data.head()}")
print(f"columns : {movie_data.columns}")


# 배급사 정리
# 우선 대형배급사 + etc로 나눔 (이후 아래에서 원핫인코딩)
movie_data['distribution_companies'] = movie_data['distribution_companies'].apply(lambda x: categorize_companies(x))
movie_data['type'] = movie_data['type'].apply(lambda x: standardize_type_name(x))
movie_data['nation'] = movie_data['nation'].apply(lambda x: standardize_nation_name(x))
movie_data['rating'] = movie_data['rating'].apply(lambda x: standardize_ratings(x))

# One-Hot Encoding
columns_to_encode = [
    ('genre', 'genre'),
    ('distribution_companies', 'distributor'),
    ('prod_stat', 'prod_stat'),
    ('type', 'type'),
    ('nation', 'nation'),
    ('rating', 'rating')
]

# Apply one-hot encoding to all specified columns
for column, prefix in columns_to_encode:
    movie_data = one_hot_encode_column(movie_data, column, prefix)
    print(f"Columns after one-hot encoding {column}: {movie_data.columns}")


# 안 중요한 칼럼 드롭
movie_data.drop(columns=['production_companies','importation_companies'], inplace = True)

print(f"columns after one-hot encoding & dropping unnecessary ones:{movie_data.columns}")

# 개봉일자 주차로 바꿈 # int64
movie_data["open_week"] = pd.to_datetime(movie_data['open_date']).dt.year * 100 + pd.to_datetime(movie_data['open_date']).dt.isocalendar().week
movie_data['open_week'] = movie_data['open_week'].astype(str)

#박스오피스 데이터 조정
#박스오피스 데이터 dataframe으로 변경
data_ck = pd.DataFrame(data_ck)
data_cf = pd.DataFrame(data_cf)
data_nk = pd.DataFrame(data_nk)
data_nf = pd.DataFrame(data_nf)

# 박스 오피스 데이터 하나로 묶음 : box_office_data
movie_types = [
    (data_ck, 'ck'),
    (data_cf, 'cf'),
    (data_nk, 'nk'),
    (data_nf, 'nf')
]
box_office_data = pd.concat([data.assign(source=name) for data, name in movie_types], ignore_index=True)

#dropna
box_office_data = box_office_data[(box_office_data != '').all(axis=1)]

# 박스 오피스 데이터 칼럼 붙임
box_office_data.rename(columns={0: "Date", 1: "FirstBOWeek", 2: "Rank", 3: "movie_code", 4: "movie_name", 5: "open_date",
                                6: "SalesAmount", 7: "SalesShare", 8: "SalesInten", 9: "SalesChange", 10: "SalesAcc",
                                11: "AudienceCount", 12: "AudienceInten", 13: "AudienceChange", 14: "AudienceAcc",
                                15: "ScreenCount", 16: "ShowCount", 17: "BOCategory"}, inplace = True)

# change data formats
box_office_data["Date"] = pd.to_datetime(box_office_data["Date"]).dt.date  # change dates to datetime format
box_office_data[["Rank", "SalesAmount", "SalesShare", "SalesInten",  # change numbers to int
      "SalesChange", "SalesAcc", "AudienceCount", "AudienceInten",
      "AudienceChange", "AudienceAcc", "ScreenCount", "ShowCount"]] = box_office_data[["Rank", "SalesAmount", "SalesShare",
                                                                            "SalesInten", "SalesChange", "SalesAcc",
                                                                            "AudienceCount", "AudienceInten",
                                                                            "AudienceChange", "AudienceAcc",
                                                                            "ScreenCount", "ShowCount"]].apply(
    pd.to_numeric)

box_office_data = box_office_data.sort_values('Date')

# grab accumulated audience
# Group by movie_name and get the maximum accumulated_audience and the corresponding revenue
box_office_summary = box_office_data.groupby('movie_code').agg({
    'FirstBOWeek': 'first',  # We take the first year_week as we're interested in the opening week
    'SalesAmount': 'first',    # We take the first revenue as it corresponds to the opening week
    'SalesShare' : 'first',
    'AudienceCount' : 'first',
    'AudienceAcc': 'max'  # We take the max accumulated audience
}).reset_index()


movie_data = movie_data.merge(
    box_office_summary[['FirstBOWeek', 'movie_code', 'SalesAmount', 'SalesShare', 'AudienceCount', 'AudienceAcc']],
    left_on=['movie_code'],
    right_on=['movie_code'],
    how='left'
)

# extract final audience count
y = movie_data['AudienceAcc']
X = movie_data.drop(columns = ['AudienceAcc'])
X = X.drop(columns = ['prod_year','movie_code','movie_name','movie_name_en','open_date','open_week','FirstBOWeek'])
X['runtime'] = X['runtime'].astype(int)

print(X[0:1])
print(y[0:1])
'''---------------------------------------------------------'''
'''data exploration'''
'''decision tree'''
# decision tree
tree = DecisionTreeRegressor(max_depth=3, random_state=0)
tree.fit(X, y)

# plot decision tree
plt.figure(figsize = (20,10))
plot_tree(tree, feature_names = X.columns.tolist(), filled= True, rounded = True)
plt.tight_layout()
plt.savefig("figures/Data_exploration_tree.png")
#plt.show()
# sample count is too small
'''---------------------------------------------------------'''
'''Modelling Prep'''
'''train_test_split'''
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=0)

'''---------------------------------------------------------'''
'''Modelling'''
'''LinearRegression'''
# fit linear regression with audience count only
X_train_audience = pd.DataFrame(X_train['AudienceCount'])
X_test_audience = pd.DataFrame(X_test['AudienceCount'])
lin = LinearRegression()
lin.fit(X_train_audience, y_train)
print(f"Simple Linear Regression Accuracy : {lin.score(X_test_audience, y_test)}")

# fit multiple linear regression
lin.fit(X_train, y_train)
print(f"Multiple Linear Regression Accuracy : {lin.score(X_test, y_test)}")

