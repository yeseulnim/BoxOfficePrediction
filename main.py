import urllib.request
import json
from datetime import date
import pandas as pd
import numpy as np

from data_prep_functions import prepare_data
from data_prep_functions import get_movie_info

'''import data'''
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

data_list = [data_cf,data_ck,data_nf,data_nk]
