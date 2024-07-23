import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from modelling_functions import get_X_and_y
from modelling_functions import regression_acc
from modelling_functions import histogram_two
from modelling_functions import randomforest_acc
from modelling_functions import xgboost_acc

import xgboost
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sklearn.metrics as metrics

from sklearn.feature_selection import SelectFromModel

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

movie_data_list = ['commercial korean.csv', 'commercial foreign.csv',
              'noncommercial korean.csv', 'noncommercial foreign.csv']

# Create two figures, one for AudienceAcc and one for AudienceCount
fig_acc, axs_acc = plt.subplots(2, 2, figsize=(10, 7))
fig_count, axs_count = plt.subplots(2, 2, figsize=(10, 7))

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
fig_acc.savefig('figures/all_movies_AudienceAcc.png')
fig_count.savefig('figures/all_movies_AudienceCount.png')

#plt.show()
'''---------------------------------------------------------'''

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

bins = np.linspace(0, 13000000, 20)

for ax, column in zip([ax1, ax2], ['AudienceAcc', 'AudienceCount']):
    data_list = []
    for file in movie_data_list:
        data = pd.read_csv("data/" + file)
        data_list.append(data[column])

    ax.hist(data_list, bins=bins, stacked=True, density=True,
            label=[f.split('.')[0] for f in movie_data_list],
            color=colors)

    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    if column == 'AudienceAcc':
        ax.set_title('Distribution of Final Audience')
    else:
        ax.set_title('Distribution of First Day Audience')

plt.tight_layout()
fig.savefig('figures/stacked_movie_data_histograms.png')
#plt.show()


'''---------------------------------------------------------'''


'''-----------------------------------------------------------'''
'''Gridsearch on XGBoost and Random Forest'''
'''
X, y = get_X_and_y('movie_data.csv')
def modelfit(pip_xgb, grid_param_xgb, x, y) :
    gs_xgb = (GridSearchCV(estimator=pip_xgb,
                        param_grid=grid_param_xgb,
                        cv=4,
                        # scoring='neg_mean_squared_error',
                        scoring='neg_root_mean_squared_error',
                        n_jobs=-1,
                        verbose=10))

    gs_xgb = gs_xgb.fit(x, y)
    print('Train Done.')

    #Predict training set:
    y_pred = gs_xgb.predict(x)

    #Print model report:
    print("\nModel Report")
    print("\nCV 결과 : ", gs_xgb.cv_results_)
    print("\n베스트 정답률 : ", gs_xgb.best_score_)
    print("\n베스트 파라미터 : ", gs_xgb.best_params_)


pip_xgb1 = Pipeline([('scl', StandardScaler()),
    ('reg', RandomForestRegressor())])
grid_param_xgb1 = {
    'reg__max_depth' : [5, 10, 15],
    'reg__criterion' : ['squared_error', 'friedman_mse', 'poisson'],
    'reg__min_samples_leaf' : [3, 5, 7],
    'reg__min_samples_split' : [3, 5, 7]
}

modelfit(pip_xgb1, grid_param_xgb1, X, y)
'''



'''-----------------------------------------------------------'''
'''Modelling'''
'''Linear regression'''

movie_data_list.insert(0, 'movie_data.csv')


simple_regression_result = []
multiple_regression_result = []
rf_regression_result = []
xgb_regression_results = []

for file in movie_data_list:
    X, y = get_X_and_y(file)

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
    xgb.fit(X_train, y_train)
    fig, ax = plt.subplots(figsize=(14, 6))
    xgboost.plot_importance(xgb, ax=ax, importance_type='gain',
                            title=f'feature importance : {file.split('.')[0]} films', max_num_features=10)
    plt.savefig(f'figures/feature_importance_{file.split('.')[0]}.png')


    simple_regression_result.append(simple_result)
    multiple_regression_result.append(multiple_result)
    rf_regression_result.append(rf_result)
    xgb_regression_results.append(xgb_result)

    model_name_list = ["lin_model_1", "lin_model_2", "rf_model", "xgb_model"]
    model_list = [lin_model_1, lin_model_2, rf_model, xgb_model]

    for model_name, model in zip(model_name_list, model_list):
        joblib.dump(model, f'models/{file.split('.')[0]}_{model_name}.pkl')

'''---------------------------------------------------------'''
'''Plot'''
'''Linear Regression'''
histogram_two(simple_regression_result, multiple_regression_result,
              'First Weekend Audience', 'Multiple Variables',
              'Prediction Accuracy for Final Audience Numbers (Linear Regression)',
              'regression_result.png')


'''Plot'''
'''RF & XGBoost Regression'''
histogram_two(rf_regression_result, xgb_regression_results,
              'Random Forest', 'XGBoost',
              'Prediction Accuracy for Final Audience Numbers (Ensemble)',
              'regression_result_ensemble.png')


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
df.to_csv('data/regression_results.csv')

print("CSV file 'regression_results.csv' has been created.")

'''---------------------------------------------------------'''
'''Save Models'''
models = []

