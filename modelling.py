from datetime import date
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


from sklearn.model_selection import train_test_split
from modelling_functions import get_X_and_y
from modelling_functions import shallow_decision_tree
from modelling_functions import regression_acc
from modelling_functions import histogram_two
from modelling_functions import randomforest_acc
from modelling_functions import kmeans_clustering


import matplotlib.pyplot as plt

'''---------------------------------------------------------'''
'''Display setting'''
# display all columns
pd.set_option('display.max_seq_items', None)
pd.set_option('display.max_columns', None)
'''------------------------------------------------------------------------------------'''
#Set Font for Korean letters
plt.rcParams['font.sans-serif']=['Noto Sans CJK KR']
plt.rcParams['axes.unicode_minus']=False
'''---------------------------------------------------------'''

'''---------------------------------------------------------'''
# Linear regression

movie_data_list = ['movie_data.csv', 'movie_data_ck.csv', 'movie_data_cf.csv',
              'movie_data_nk.csv', 'movie_data_nf.csv']


simple_regression_result = []
multiple_regression_result = []
rf_regression_result = []
for file in movie_data_list:
    X, y = get_X_and_y(file)

    '''train_test_split'''
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=0)
    '''linear regression'''
    simple_result, multiple_result = regression_acc(X_train, X_test, y_train, y_test)
    rf_result = randomforest_acc(X_train, X_test, y_train, y_test)
    simple_regression_result.append(simple_result)
    multiple_regression_result.append(multiple_result)
    rf_regression_result.append(rf_result)

'''---------------------------------------------------------'''
'''Plot'''
'''Linear Regression'''
histogram_two(simple_regression_result, multiple_regression_result, 'regression_result.png')


print(rf_regression_result)


'''---------------------------------------------------------'''
'''Clustering'''
movie_data_sublist = ['movie_data_ck.csv', 'movie_data_cf.csv',
              'movie_data_nk.csv', 'movie_data_nf.csv']


for data in movie_data_sublist:
    km = kmeans_clustering(data)
    try:
        kmstack = pd.concat([km, kmstack], ignore_index = True)
    except:
        kmstack = km

kmstack.to_csv('centroids.csv')