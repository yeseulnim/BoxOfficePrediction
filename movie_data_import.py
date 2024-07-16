import urllib.request
import json
from datetime import date
import pandas as pd
import numpy as np

from data_prep_functions import get_movie_info



'''get movie information'''
'''get moviecode'''

data_cf = pd.read_csv('data/data_cf.csv')
data_ck = pd.read_csv('data/data_ck.csv')
data_nf = pd.read_csv('data/data_nf.csv')
data_nk = pd.read_csv('data/data_nk.csv')


moviecodes = pd.concat([data_cf["MovieCode"],data_ck["MovieCode"],data_nf["MovieCode"],data_nk["MovieCode"]], ignore_index = True)
moviecodes = list(set(moviecodes)) # 4514 movies
moviecodes_1, moviecodes_2 = moviecodes[:len(moviecodes)//2], moviecodes[len(moviecodes)//2:]
print(len(moviecodes_1))
print(len(moviecodes_2))

get_movie_info(moviecodes_2,"movie_info_2.json")