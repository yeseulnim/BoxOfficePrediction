import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from modelling_functions import get_X_and_y
from modelling_functions import regression_acc
from modelling_functions import histogram_two_for_clusters
from modelling_functions import histogram_four_for_clusters
from modelling_functions import randomforest_acc
from modelling_functions import xgboost_acc
from modelling_functions import kmeans_clustering

import xgboost
from xgboost import XGBRegressor

import matplotlib.pyplot as plt

import joblib


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
'''Exploratory Data Analysis'''

movie_data_list = ['commercial korean.csv', 'commercial foreign.csv']

'''---------------------------------------------------------'''
'''Clustering'''

data_directory = 'data'
figure_directory = 'figures'

# Initialize an empty list to store the best k-means results
best_kmeans_results = []

# Perform k-means clustering for each file
for data in movie_data_list:
    km = kmeans_clustering(data_directory, data)
    best_kmeans_results.append(km)

# Combine all results into a single DataFrame
kmstack = pd.concat(best_kmeans_results, ignore_index=True)

# Save the combined results
kmstack.to_csv('data/best_centroids.csv', index=False)


'''------------------------------------------------------'''



'''------------------------------------------------------------'''
'''Regression, Random Forest & XGBoost for each subcategory'''


movie_data_list = [
                   'commercial korean_cluster_0.csv', 'commercial korean_cluster_1.csv',
                   'commercial korean_cluster_2.csv',
                   'commercial foreign_cluster_0.csv','commercial foreign_cluster_1.csv',
                   'commercial foreign_cluster_2.csv']

# Create two figures, one for AudienceAcc and one for AudienceCount
fig_acc, axs_acc = plt.subplots(2, 3, figsize=(10, 7))
fig_count, axs_count = plt.subplots(2, 3, figsize=(10, 7))

# Flatten the 2x3 arrays to make indexing easier
axs_acc = axs_acc.flatten()
axs_count = axs_count.flatten()

for i, file in enumerate(movie_data_list):
    data = pd.read_csv("data/" + file)
    title = file.split('.')[0]

    # Plot AudienceAcc
    axs_acc[i].hist(data['AudienceAcc'], bins=20)
    axs_acc[i].set_title(f"{title}")

    # Plot AudienceCount
    axs_count[i].hist(data['AudienceCount'], bins=20)
    axs_count[i].set_title(f"{title}")

# Remove the empty sixth subplots

# Add overall titles
fig_acc.suptitle("Final Audience Count", fontsize=16)
fig_count.suptitle("First Day Audience Count", fontsize=16)

# Adjust layout and save figures
fig_acc.savefig('figures/clustered_all_movies_AudienceAcc.png')
fig_count.savefig('figures/clustered_all_movies_AudienceCount.png')

#plt.show()
'''---------------------------------------------------------'''


'''---------------------------------------------------------'''
'''Modelling'''
'''Linear regression'''

simple_regression_result = []
multiple_regression_result = []
rf_regression_result = []
xgb_regression_results = []

for file in movie_data_list:
    X, y = get_X_and_y(file)

    X.drop(columns = ['movie_code', 'movie_name'], inplace = True)

    '''train_test_split'''
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=0)

    '''linear regression'''
    lin_model_1, lin_model_2, simple_result, multiple_result = regression_acc(X_train, X_test, y_train, y_test)
    '''random forest'''
    rf_model, rf_result = randomforest_acc(X_train, X_test, y_train, y_test)

    '''xgboost'''
    xgb_model, xgb_result = xgboost_acc(X_train, X_test, y_train, y_test)

    '''xgboost feature importance'''
    xgb = XGBRegressor()
    xgb.fit(X_train,y_train)
    fig, ax = plt.subplots(figsize=(14, 6))
    xgboost.plot_importance(xgb,ax=ax, importance_type='gain',title=f'feature importance : {file.split('.')[0]} films',max_num_features=10)
    plt.savefig(f'figures/clustered_feature_importance_{file.split('.')[0]}.png')

    simple_regression_result.append(simple_result)
    multiple_regression_result.append(multiple_result)
    rf_regression_result.append(rf_result)
    xgb_regression_results.append(xgb_result)

    model_name_list = ["lin_model_1", "lin_model_2", "rf_model", "xgb_model"]
    model_list = [lin_model_1, lin_model_2, rf_model, xgb_model]

    for model_name, model in zip(model_name_list, model_list):
        joblib.dump(model, f'models/cluster_{file.split('.')[0]}_{model_name}.pkl')

'''---------------------------------------------------------'''
'''Plot'''
'''Linear Regression'''
histogram_two_for_clusters(simple_regression_result, multiple_regression_result,
              'First Weekend Audience', 'Multiple Variables',
              'Prediction Accuracy for Final Audience Numbers (Linear Regression)',
              'clustered_regression_result.png')

'''---------------------------------------------------------'''
'''Plot'''
'''RF & XGBoost Regression'''
histogram_two_for_clusters(rf_regression_result, xgb_regression_results,
              'Random Forest', 'XGBoost',
              'Prediction Accuracy for Final Audience Numbers (Ensemble)',
              'clustered_regression_result_ensemble.png')

'''Plot'''
'''all four'''
histogram_four_for_clusters(simple_regression_result,multiple_regression_result,rf_regression_result,xgb_regression_results,
               'First Weekend Audience', 'Multiple Regression','Random Forest','XGBoost',
               'Prediction Accuracy for Final Audience Numbers',
               'clustered_regression_result_all')


'''---------------------------------------------------------'''
'''Save Accuracy on CSV'''
data = {
    'Simple Regression': simple_regression_result,
    'Multiple Regression': multiple_regression_result,
    'RF Regression': rf_regression_result,
    'XGB Regression': xgb_regression_results
}

# Create a DataFrame
df = pd.DataFrame(data)

df = df.transpose()

# Write to CSV
df.to_csv('data/clustered_regression_results.csv')

print("CSV file 'clustered_regression_results.csv' has been created.")










