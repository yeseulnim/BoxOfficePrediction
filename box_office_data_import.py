import urllib.request
import json
from datetime import date
from data_prep_functions import get_weekend_box_office
from data_prep_functions import prepare_data

'''save data on json'''
# 2010 - 2019
# end_date = date(2020,2,29) # 2010 ~ 2019 + 2020 Jan
# num_days = 3712 # 3650 + 31 (Jan) + 29 (Feb) + 2 (leap year)

# 2023
end_date = date(2023,1,31)
num_days = 395 # 365 + 31 (Jan) + 29 (Feb)


get_weekend_box_office(end_date,numdays= num_days,
                       commercial = "Y", nation = "F", filename = "commercial_foreign.json")
get_weekend_box_office(end_date,numdays= num_days,
                       commercial = "Y", nation = "K", filename = "commercial_korean.json")
get_weekend_box_office(end_date,numdays= num_days,
                       commercial = "N", nation = "F", filename = "noncommercial_foreign.json")
get_weekend_box_office(end_date,numdays= num_days,
                       commercial = "N", nation = "K", filename = "noncommercial_korean.json")


'''import data from json'''
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

'''prepare data'''
data_cf = prepare_data(data_cf)
data_ck = prepare_data(data_ck)
data_nf = prepare_data(data_nf)
data_nk = prepare_data(data_nk)

'''save data on CSV'''
data_cf.to_csv("data/data_cf.csv", sep = ",", na_rep = "NaN")
data_ck.to_csv("data/data_ck.csv", sep = ",", na_rep = "NaN")
data_nf.to_csv("data/data_nf.csv", sep = ",", na_rep = "NaN")
data_nk.to_csv("data/data_nk.csv", sep = ",", na_rep = "NaN")

