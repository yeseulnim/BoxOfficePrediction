import json

import pandas as pd
from sklearn.preprocessing import RobustScaler

from data_prep_functions import extract_movie_info
from data_prep_functions import categorize_companies
from data_prep_functions import standardize_type_name
from data_prep_functions import standardize_nation_name
from data_prep_functions import standardize_ratings
from data_prep_functions import one_hot_encode_column
from score_scrap_function import get_star_rating
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
'''-----------------------------------------------------'''

#read from csv
movie_1 = pd.read_csv("data/movie_data_1.csv")
movie_2 = pd.read_csv("data/movie_data_2.csv")

movie = pd.concat([movie_1, movie_2])

movie['movie_code'] = movie['movie_code'].astype(str)


'''-----------------------------------------------------'''
movie_info = movie_info_1 + movie_info_2
movie_data = pd.DataFrame([extract_movie_info(movie) for movie in movie_info])
print(movie_data.head())
movie_data['movie_code'] = movie_data['movie_code'].astype(str)


movie_data = movie_data.merge(
    movie[['movie_code', 'star_ratings']],
    on='movie_code',
    how='left'
)

'''-----------------------------------------------------'''

# print stats for movie_data
print(f"Total movie count: {len(movie_data)}")
print(f"samples: {movie_data.head()}")
print(f"columns : {movie_data.columns}")

'''-----------------------------------------------------'''
# 개봉일자 주차로 바꿈 # int64
movie_data["open_week"] = pd.to_datetime(movie_data['open_date']).dt.year * 100 + pd.to_datetime(movie_data['open_date']).dt.isocalendar().week
movie_data['open_week'] = movie_data['open_week'].astype(str)
'''-----------------------------------------------------'''
movie_data['movie_code'] = movie_data['movie_code'].astype(str)

'''-----------------------------------'''
'''Box Office data 조정'''
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
'''-----------------------------------'''
'''drop na'''

box_office_data = box_office_data[(box_office_data != '').all(axis=1)]
movie_data = movie_data[(movie_data != '').all(axis=1)]
'''-----------------------------------'''
'''drop rating 0s'''
movie_data = movie_data[(movie_data['star_ratings'] != 0)]


'''-----------------------------------'''

# 박스 오피스 데이터 칼럼 붙임
box_office_data.rename(columns={0: "Date", 1: "FirstBOWeek", 2: "Rank", 3: "movie_code", 4: "movie_name", 5: "open_date",
                                6: "SalesAmount", 7: "SalesShare", 8: "SalesInten", 9: "SalesChange", 10: "SalesAcc",
                                11: "AudienceCount", 12: "AudienceInten", 13: "AudienceChange", 14: "AudienceAcc",
                                15: "ScreenCount", 16: "ShowCount", 'source': "BOCategory"}, inplace = True)

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

print(f'Box Office data columns: {box_office_data.columns}')

# grab accumulated audience
# Group by movie_name and get the maximum accumulated_audience and the corresponding revenue
box_office_summary = box_office_data.groupby('movie_code').agg({
    'FirstBOWeek': 'first',  # We take the first year_week as we're interested in the opening week
    'SalesAmount': 'first',    # We take the first revenue as it corresponds to the opening week
    'SalesShare' : 'first',
    'AudienceCount' : 'first',
    'AudienceAcc': 'max',  # We take the max accumulated audience
    'BOCategory': 'first'
}).reset_index()


'''-----------------------------------------------------'''
'''Movie Data에 Box Office Data 자료 합침'''
movie_data = movie_data.merge(
    box_office_summary[['FirstBOWeek', 'movie_code', 'SalesAmount', 'SalesShare', 'AudienceCount', 'AudienceAcc', 'BOCategory']],
    left_on=['movie_code'],
    right_on=['movie_code'],
    how='left'
)

'''-----------------------------------------------------'''
'''Movie Data 추가조정'''
# 안 중요한 칼럼 드롭
movie_data.drop(columns=['production_companies','importation_companies'], inplace = True)
movie_data.drop(columns = ['prod_year','movie_code','movie_name','movie_name_en','open_date','open_week','FirstBOWeek'], inplace = True)

# Pre One-Hot Encoding
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
    ('rating', 'rating'),
    ('BOCategory', 'BOCategory')
]

# Apply one-hot encoding to all specified columns
for column, prefix in columns_to_encode:
    movie_data = one_hot_encode_column(movie_data, column, prefix)
    print(f"Columns after one-hot encoding {column}: {movie_data.columns}")

print(f"columns after one-hot encoding & dropping unnecessary ones:{movie_data.columns}")

'''--------------------------------------------------'''

# Data Normalization
rbs = RobustScaler()

movie_data['runtime'] = rbs.fit_transform(pd.DataFrame(movie_data['runtime']))
#movie_data['investor_count'] = rbs.fit_transform(pd.DataFrame(movie_data['staff_count']))
#movie_data['company_count'] = rbs.fit_transform(pd.DataFrame(movie_data['staff_count']))
#movie_data['staff_count'] = rbs.fit_transform(pd.DataFrame(movie_data['staff_count']))
movie_data['SalesAmount'] = rbs.fit_transform(pd.DataFrame(movie_data['SalesAmount']))
#movie_data['SalesShare'] = rbs.fit_transform(pd.DataFrame(movie_data['SalesShare']))
#movie_data['AudienceCount'] = rbs.fit_transform(pd.DataFrame(movie_data['AudienceCount']))
#movie_data['AudienceAcc'] = rbs.fit_transform(pd.DataFrame(movie_data['AudienceAcc']))


'''--------------------------------------------------'''
'''Save to CSV'''
movie_data.to_csv('data/movie_data.csv')

# korean commercial films
movie_data_ck = movie_data[movie_data['BOCategory_ck'] == 1]
movie_data_ck.drop(columns = ['BOCategory_ck','BOCategory_cf','BOCategory_nk','BOCategory_nf'])
movie_data_ck.to_csv('data/movie_data_ck.csv')

# foreign commercial films
movie_data_cf = movie_data[movie_data['BOCategory_cf'] == 1]
movie_data_cf.drop(columns = ['BOCategory_ck','BOCategory_cf','BOCategory_nk','BOCategory_nf'])
movie_data_cf.to_csv('data/movie_data_cf.csv')

# korean noncommercial films
movie_data_nk = movie_data[movie_data['BOCategory_nk'] == 1]
movie_data_nk.drop(columns = ['BOCategory_ck','BOCategory_cf','BOCategory_nk','BOCategory_nf'])
movie_data_nk.to_csv('data/movie_data_nk.csv')

# foreign noncommercial films
movie_data_nf = movie_data[movie_data['BOCategory_nf'] == 1]
movie_data_nf.drop(columns = ['BOCategory_ck','BOCategory_cf','BOCategory_nk','BOCategory_nf'])
movie_data_nf.to_csv('data/movie_data_nf.csv')

'''-------------------------------------------------------------------------------------------'''
