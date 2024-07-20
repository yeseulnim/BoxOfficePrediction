import json
import pandas as pd

from data_prep_functions import extract_movie_info
from score_scrap_function import get_star_rating

'''---------------------------------------------------------'''
'''Display setting'''
# display all columns
pd.set_option('display.max_seq_items', None)
pd.set_option('display.max_columns', None)
'''---------------------------------------------------'''

with open('data/movie_info_1.json') as f:
    movie_info_1 = json.load(f)
f.close()

with open('data/movie_info_2.json') as f:
    movie_info_2 = json.load(f)
f.close()


'''------------------------------------------------------'''
'''Movie data 조정 1'''

'''------------------------------------------------------'''
'''Get Star Ratings'''

movie_infos = [movie_info_1, movie_info_2] # 향후 movie_info_1 추가한 코드로 수정
for ind, info in enumerate(movie_infos):

    data = pd.DataFrame([extract_movie_info(movie) for movie in info])
    data = data[(data != '').all(axis=1)]


    '''-----------------------------------------------------'''
    '''Get Star Ratings'''

    star_ratings = []
    for idx, row in data.iterrows():
        movie_name = row['movie_name']
        movie_year = row['prod_year']
        movie_nation = row['nation']
        star_rating = get_star_rating(movie_name,movie_year,movie_nation)
        star_ratings.append(star_rating)

    data['star_ratings'] = star_ratings
    '''-----------------------------------------------------'''
    '''Save into CSV'''
    data.to_csv(f"data/movie_data_{ind}.csv")

'''--------------------------------------------------------------------------'''