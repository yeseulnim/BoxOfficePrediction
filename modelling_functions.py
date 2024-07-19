import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans


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
    # fit multiple linear regression
    lin.fit(X_train, y_train)
    print(f"Multiple Linear Regression Accuracy : {lin.score(X_test, y_test)}")
    multiple_result = lin.score(X_test, y_test)
    return simple_result, multiple_result

def histogram_two(list1, list2, filename):
    plt.figure(figsize = (12,4))
    X_ticks = ['All Movies','Commercial Korean', 'Commercial Foreign','Non-Commercial Korean','Non-Commercial Foreign']

    X_axis = np.arange(len(X_ticks))

    plt.bar(X_axis - 0.2, list1, 0.4, label='Simple LinReg')
    plt.bar(X_axis + 0.2, list2, 0.4, label='Multiple LinReg')

    plt.xticks(X_axis, X_ticks)
    plt.xlabel("Groups")
    plt.ylabel("Accuracy")
    plt.title("First Weekend Audience : Relevance to Final Audience")
    plt.legend()
    plt.tight_layout
    plt.savefig(f"figures/{filename}")
    #plt.show()


def randomforest_acc(X_train, X_test,y_train,y_test):
    X_train_audience = pd.DataFrame(X_train['AudienceCount'])
    X_test_audience = pd.DataFrame(X_test['AudienceCount'])
    ran = RandomForestRegressor(n_estimators=200, criterion="squared_error", min_samples_leaf=5, min_samples_split=5, max_depth=5)
    ran.fit(X_train_audience, y_train)
    print(f"Random Forest Accuracy : {ran.score(X_test_audience, y_test)}")

    return ran.score(X_test_audience, y_test)


def kmeans_clustering(filename):
    data = pd.read_csv(f'data/{filename}')
    columns = data.columns
    kme = KMeans(n_clusters = 2)
    kme.fit(data)
    clusters = pd.DataFrame(kme.cluster_centers_)
    clusters.columns = columns
    return clusters



