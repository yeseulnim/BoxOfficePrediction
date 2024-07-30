import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

import xgboost

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.cluster import KMeans
from kneed import KneeLocator

def get_X_and_y(filename):
    # run data_prep
    # read the resulting csv file
    movie_data = pd.read_csv(f'data/{filename}')
    print(f"Movie Count : {len(movie_data)}")

    '''---------------------------------------------------------'''
    # Round AudienceAcc to 10,000s
    # movie_data['AudienceAcc'] = movie_data['AudienceAcc'].round(4)
    # print(movie_data['AudienceAcc'].head())
    # Above is cancelled as regression accuracy goes down(!)


    # extract final audience count
    y = movie_data['AudienceAcc']
    X = movie_data.drop(columns = ["Unnamed: 0", 'AudienceAcc' ,'SalesShare'])
    X['runtime'] = X['runtime'].astype(int)
    return X, y



def histogram_two(list1, list2, label1, label2, title, x_ticks, filename):
    plt.figure(figsize = (12,4))
    X_ticks = x_ticks

    X_axis = np.arange(len(X_ticks))

    plt.bar(X_axis - 0.2, list1, 0.4, label=label1)
    plt.bar(X_axis + 0.2, list2, 0.4, label=label2)

    plt.xticks(X_axis, X_ticks)
    plt.xlabel("Groups")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout
    plt.savefig(f"figures/{filename}")
    #plt.show()


def histogram_four(list1, list2, list3, list4, label1, label2, label3, label4, x_ticks, title, filename):
    plt.figure(figsize = (12,5))
    X_ticks = x_ticks

    X_axis = np.arange(len(X_ticks))

    bar_width = 0.2
    plt.bar(X_axis - 1.5*bar_width, list1, bar_width, label=label1)
    plt.bar(X_axis - 0.5*bar_width, list2, bar_width, label=label2)
    plt.bar(X_axis + 0.5*bar_width, list3, bar_width, label=label3)
    plt.bar(X_axis + 1.5*bar_width, list4, bar_width, label=label4)

    plt.xticks(X_axis, X_ticks)
    plt.xlabel("Groups")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout
    plt.savefig(f"figures/{filename}")
    #plt.show()

def randomforest_gridsearch_cv(X, y, cv=5):
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'criterion': ['poisson', 'squared_error'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'max_depth': [10, 15, 20, None]
    }

    # Create a base model
    rf = RandomForestRegressor()

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=cv, n_jobs=-1, verbose=2, scoring='r2')

    # Fit the grid search to the data
    grid_search.fit(X, y)

    # Print the best parameters and score
    print("Best parameters found: ", grid_search.best_params_)
    print(f"Best cross-validation R2 score: {grid_search.best_score_:.4f}")

    # Get the best model
    best_model = grid_search.best_estimator_

    return best_model, grid_search.best_score_



def xgboost_gridsearch_cv(X, y, cv=5):
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }

    # Create a base model
    xgb_model = xgboost.XGBRegressor(objective='reg:squarederror', random_state=42)

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                               cv=cv, n_jobs=-1, verbose=2, scoring='r2')

    # Fit the grid search to the data
    grid_search.fit(X, y)

    # Print the best parameters and score
    print("Best parameters found: ", grid_search.best_params_)
    print(f"Best cross-validation R2 score: {grid_search.best_score_:.4f}")

    # Get the best model
    best_model = grid_search.best_estimator_

    return best_model, grid_search.best_score_



def regression_gridsearch_cv(X, y, cv=5):
    # Simple Linear Regression
    X_audience = pd.DataFrame(X['AudienceCount'])

    # Define parameter grid for Ridge and Lasso (used instead of simple LinearRegression for regularization)
    param_grid_simple = {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
    }

    # Ridge Regression for simple linear regression
    ridge_simple = Ridge()
    grid_search_simple = GridSearchCV(estimator=ridge_simple, param_grid=param_grid_simple,
                                      cv=cv, n_jobs=-1, verbose=2, scoring='r2')
    grid_search_simple.fit(X_audience, y)

    print("Best parameters for Simple Linear Regression: ", grid_search_simple.best_params_)
    print(f"Best cross-validation R2 score for Simple Linear Regression: {grid_search_simple.best_score_:.4f}")

    # Multiple Linear Regression
    param_grid_multiple = {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
        'fit_intercept': [True, False]
    }

    # Lasso Regression for multiple linear regression (handles multicollinearity)
    lasso_multiple = Lasso()
    grid_search_multiple = GridSearchCV(estimator=lasso_multiple, param_grid=param_grid_multiple,
                                        cv=cv, n_jobs=-1, verbose=2, scoring='r2')
    grid_search_multiple.fit(X, y)

    print("Best parameters for Multiple Linear Regression: ", grid_search_multiple.best_params_)
    print(f"Best cross-validation R2 score for Multiple Linear Regression: {grid_search_multiple.best_score_:.4f}")

    # Get the best models
    best_model_simple = grid_search_simple.best_estimator_
    best_model_multiple = grid_search_multiple.best_estimator_

    return best_model_simple, best_model_multiple, grid_search_simple.best_score_, grid_search_multiple.best_score_


def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return rmse, mae, r2

'''
def randomforest_acc(X_train, X_test,y_train,y_test)
    ran = RandomForestRegressor(n_estimators=300, criterion="poisson", min_samples_leaf=2, min_samples_split=5, max_depth=17)

    ran.fit(X_train, y_train)
    print(f"Random Forest Accuracy : {ran.score(X_test, y_test)}")

    return ran, ran.score(X_test, y_test)

def xgboost_acc(X_train, X_test,y_train,y_test):
    xgb = xgboost.XGBRegressor()
    xgb.fit(X_train, y_train)
    print(f"XGBoost Accuracy : {xgb.score(X_test, y_test)}")

    return xgb, xgb.score(X_test, y_test)
'''

'''
def regression_acc(X_train, X_test,y_train,y_test):
    X_train_audience = pd.DataFrame(X_train['AudienceCount'])
    X_test_audience = pd.DataFrame(X_test['AudienceCount'])
    lin = LinearRegression()
    lin.fit(X_train_audience, y_train)
    print(f"Simple Linear Regression Accuracy : {lin.score(X_test_audience, y_test)}")
    simple_result = lin.score(X_test_audience, y_test)

    model1 = lin

    # fit multiple linear regression
    lin2 = LinearRegression()
    lin2.fit(X_train, y_train)
    print(f"Multiple Linear Regression Accuracy : {lin2.score(X_test, y_test)}")
    multiple_result = lin2.score(X_test, y_test)

    model2 = lin2

    return model1, model2, simple_result, multiple_result
'''
def histogram_two_for_clusters(list1, list2, label1, label2, title, filename):
    plt.figure(figsize = (12,4))
    X_ticks = ['C Korean - 1', 'C Korean - 2', 'C Korean - 3',
               'C Foreign - 1', 'C Foreign - 2', 'C Foreign - 3']

    X_axis = np.arange(len(X_ticks))

    plt.bar(X_axis - 0.2, list1, 0.4, label=label1)
    plt.bar(X_axis + 0.2, list2, 0.4, label=label2)

    plt.xticks(X_axis, X_ticks)
    plt.xlabel("Groups")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout
    plt.savefig(f"figures/{filename}")
    #plt.show()

def histogram_four_for_clusters(list1, list2, list3, list4, label1, label2, label3, label4, title, filename):
    plt.figure(figsize = (12,5))
    X_ticks = ['C Korean - 1', 'C Korean - 2', 'C Korean - 3',
               'C Foreign - 1', 'C Foreign - 2', 'C Foreign - 3']

    X_axis = np.arange(len(X_ticks))

    bar_width = 0.2
    plt.bar(X_axis - 1.5*bar_width, list1, bar_width, label=label1)
    plt.bar(X_axis - 0.5*bar_width, list2, bar_width, label=label2)
    plt.bar(X_axis + 0.5*bar_width, list3, bar_width, label=label3)
    plt.bar(X_axis + 1.5*bar_width, list4, bar_width, label=label4)

    plt.xticks(X_axis, X_ticks)
    plt.xlabel("Groups")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout
    plt.savefig(f"figures/{filename}")
    #plt.show()


def kmeans_clustering(data_directory, file_name):
    figure_directory = 'figures'
    # Construct the full file path
    file_path = os.path.join(data_directory, file_name)

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Preprocess the data
    columns_to_drop = ['AudienceAcc','movie_code', 'movie_name']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    X = df.drop(columns_to_drop, axis=1, errors='ignore')

    # Perform elbow method
    inertias = []
    k_range = range(1, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title(f'Elbow Method for Optimal k - {file_name}')
    plt.savefig(os.path.join(figure_directory, f'elbow_graph_{file_name.split(".")[0]}.png'))
    plt.close()

    cluster_size = 3

    kmeans = KMeans(n_clusters=cluster_size, random_state=42)
    cluster_labels = kmeans.fit_predict(X)


    df['Cluster'] = cluster_labels

    # Save movies in each cluster to separate CSV files
    for i in range(cluster_size):
        cluster_df = df[df['Cluster'] == i]
        output_filename = f"{file_name.split('.')[0]}_cluster_{i}.csv"
        output_path = os.path.join(data_directory, output_filename)
        cluster_df.to_csv(output_path, index=False)

    # Get cluster centers and add to dataframe
    cluster_centers = kmeans.cluster_centers_
    centroids_df = pd.DataFrame(cluster_centers, columns=X.columns)
    centroids_df['file_name'] = file_name
    centroids_df['cluster'] = range(len(centroids_df))

    return centroids_df
