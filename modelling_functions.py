import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost


import pickle



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
    X = movie_data.drop(columns = ['AudienceAcc' ,'SalesShare'])
    X['runtime'] = X['runtime'].astype(int)
    return X, y


def shallow_decision_tree(X,y,filename):
    '''decision tree'''
    # decision tree
    tree = DecisionTreeRegressor(max_depth=3, random_state=0)
    tree.fit(X, y)

    # plot decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(tree, feature_names=X.columns.tolist(), filled=True, rounded=True)
    plt.tight_layout()
    plt.savefig(f"figures/{filename}")
    # plt.show()
    # sample count is too small

def regression_acc(X_train, X_test,y_train,y_test):
    X_train_audience = pd.DataFrame(X_train['AudienceCount'])
    X_test_audience = pd.DataFrame(X_test['AudienceCount'])
    lin = LinearRegression()
    lin.fit(X_train_audience, y_train)
    print(f"Simple Linear Regression Accuracy : {lin.score(X_test_audience, y_test)}")
    simple_result = lin.score(X_test_audience, y_test)

    model1 = pickle.dumps(lin)

    # fit multiple linear regression
    lin.fit(X_train, y_train)
    print(f"Multiple Linear Regression Accuracy : {lin.score(X_test, y_test)}")
    multiple_result = lin.score(X_test, y_test)

    model2 = pickle.dumps(lin)

    return model1, model2, simple_result, multiple_result

def histogram_two(list1, list2, label1, label2, title, filename):
    plt.figure(figsize = (12,4))
    X_ticks = ['All Movies','Commercial Korean', 'Commercial Foreign','Non-Commercial Korean','Non-Commercial Foreign']

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


def randomforest_acc(X_train, X_test,y_train,y_test):
    ran = RandomForestRegressor(n_estimators=300, criterion="poisson", min_samples_leaf=3, min_samples_split=7, max_depth=20)
    ran.fit(X_train, y_train)
    print(f"Random Forest Accuracy : {ran.score(X_test, y_test)}")

    model = pickle.dumps(ran)

    return model, ran.score(X_test, y_test)

def xgboost_acc(X_train, X_test,y_train,y_test):
    xgb = xgboost.XGBRegressor()
    xgb.fit(X_train, y_train)
    print(f"XGBoost Accuracy : {xgb.score(X_test, y_test)}")

    model = pickle.dumps(xgb)

    return model, xgb.score(X_test, y_test)


def histogram_two_for_clusters(list1, list2, label1, label2, title, filename):
    plt.figure(figsize = (12,4))
    X_ticks = ['NC Korean - 1', 'NC Korean - 2', 'NC Korean - 3',
               'NC Foreign - 1', 'NC Foreign - 2', 'NC Foreign - 3']

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

