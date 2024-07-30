import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.stats import chi2_contingency

from sklearn.cluster import KMeans

from modelling_functions import get_X_and_y
from modelling_functions import evaluate_model
from modelling_functions import histogram_four

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
movie_data_list = ['commercial korean.csv', 'commercial foreign.csv',
                   'noncommercial korean.csv', 'noncommercial foreign.csv']
'''------------------------------------------------------------'''
'''Regression Plotting'''
'''All Movies'''
X, y = get_X_and_y('2010 - 2019/movie_data.csv')
X = pd.DataFrame(X['AudienceCount'])

# density
xy = np.vstack([X.T,y])
z = gaussian_kde(xy)(xy)

model = joblib.load(f'models/2010 - 2019/movie_data_lin_model_1.pkl')
pred = model.predict(X)

plt.figure(figsize=(4,3))
plt.title('First Weekend vs Total Audience')
plt.scatter(X, y, c = z, s = 10)
plt.plot(X,pred, color = 'red')
plt.xlabel('First Weekend Audience')
plt.ylabel('Total Audience')

plt.tight_layout()
plt.savefig('figures/regression_plot_all.png')
'''------------------------------------------------------------'''
'''C/NC, K/F'''
fig, ax = plt.subplots(1,4, figsize = (12,3))
fig.suptitle(f"First Weekend vs Total Audience")

for idx, filename in enumerate(movie_data_list):
    X, y = get_X_and_y(f'2010 - 2019/{filename}')
    X = pd.DataFrame(X['AudienceCount'])

    #density
    xy = np.vstack([X.T, y])
    z = gaussian_kde(xy)(xy)

    model = joblib.load(f'models/2010 - 2019/{filename.split('.')[0]}_lin_model_1.pkl')
    pred = model.predict(X)

    ax[idx].set_title(filename.split('.')[0])
    ax[idx].scatter(X, y, c = z, s = 10)
    ax[idx].plot(X,pred, c = 'red')

    ax[idx].set_xlabel('First Weekend Audience')
    ax[idx].set_ylabel('Total Audience')

plt.tight_layout()
plt.savefig('figures/regression_plot.png')

'''---------------------------------------------------------'''
clustered_movie_data_list = ['commercial korean_cluster_0.csv', 'commercial korean_cluster_1.csv', 'commercial korean_cluster_2.csv',
                             'commercial foreign_cluster_0.csv','commercial foreign_cluster_1.csv', 'commercial foreign_cluster_2.csv']
'''------------------------------------------------------------'''
'''C/NC, K/F'''
fig, ax = plt.subplots(2,3, figsize = (9,6))
fig.suptitle(f"First Weekend vs Total Audience")

for idx, filename in enumerate(clustered_movie_data_list):
    X, y = get_X_and_y(f'2010 - 2019/{filename}')
    X = pd.DataFrame(X['AudienceCount'])

    #density
    xy = np.vstack([X.T, y])
    z = gaussian_kde(xy)(xy)


    model = joblib.load(f'models/2010 - 2019/{filename.split('_')[0]}_lin_model_1.pkl')
    pred = model.predict(X)

    ax[idx//3][idx%3].set_title(filename.split('.')[0])
    ax[idx//3][idx%3].scatter(X, y, c = z, s = 10)
    ax[idx//3][idx%3].plot(X,pred, color = 'red')

    ax[idx//3][idx%3].set_xlabel('First Weekend Audience')
    ax[idx//3][idx%3].set_ylabel('Total Audience')

plt.tight_layout()
plt.savefig('figures/regression_plot_clustered.png')


'''------------------------------------------------------------'''
'''2023 Movies'''
model_list = ['lin_model_1.pkl', 'lin_model_2.pkl', 'rf_model.pkl', 'xgb_model.pkl']

movie_data_list.insert(0,'movie_data.csv')

prediction_scores_2023 = []
for filename in (movie_data_list):
    row = []
    for modelname in model_list:
        X, y = get_X_and_y(f'2023/{filename}')
        X.drop(columns=['movie_code', 'movie_name'], inplace=True)
        X.insert(loc=20,column = 'genre_성인물(에로)',value = 0)

        model = joblib.load(f'models/2010 - 2019/{filename.split('.')[0]}_{modelname}')
        if modelname == 'lin_model_1.pkl':
            score = model.score(pd.DataFrame(X['AudienceCount']),y)
        else:
            score = model.score(X,y)
        row.append(score)
        #print(f"{filename} {modelname} {score}")
    prediction_scores_2023.append(row)
prediction_scores_2023 = pd.DataFrame(prediction_scores_2023)
prediction_scores_2023.rename(columns = {0:'Linear Regression',1:'Multiple Regression',2:'Random Forest',3:'XGBoost'}, inplace = True)
prediction_scores_2023 = prediction_scores_2023.transpose()
prediction_scores_2023.to_csv('data/2023_prediction_scores.csv')
# prediction_scores_2023 = prediction_scores_2023.transpose()
# x_ticks = ['Commercial Korean', 'Commercial Foreign','Non-Commercial Korean','Non-Commercial Foreign']
# histogram_four(prediction_scores_2023['Linear Regression'], prediction_scores_2023['Multiple Regression'],
#                prediction_scores_2023['Random Forest'], prediction_scores_2023['XGBoost'],
#                'Simple Regression', 'Multiple Regression', 'Random Forest', 'XGBoost',
#                x_ticks, 'Prediction Accuracy for Final Audience Numbers (2023)',
#                'regression_result_2023.png')

movie_data_list = movie_data_list[1:]

'''------------------------------------------------------------'''
'''Plot 2023'''
'''C/NC, K/F'''
fig, ax = plt.subplots(1,4, figsize = (12,3))
fig.suptitle(f"First Weekend vs Total Audience (2023)")

for idx, filename in enumerate(movie_data_list):
    X, y = get_X_and_y(f'2023/{filename}')
    X = pd.DataFrame(X['AudienceCount'])

    #density
    xy = np.vstack([X.T, y])
    z = gaussian_kde(xy)(xy)

    model = joblib.load(f'models/2010 - 2019/{filename.split('.')[0]}_lin_model_1.pkl')
    pred = model.predict(X)
    print(f'{model.intercept_} {model.coef_}')

    model_2023 = joblib.load(f'models/2023/{filename.split('.')[0]}_lin_model_1.pkl')
    pred_2023 = model_2023.predict(X)
    print(f'{model_2023.intercept_} {model_2023.coef_}')


    ax[idx].set_title(filename.split('.')[0])
    ax[idx].scatter(X, y, c = z, s = 10)
    ax[idx].plot(X,pred, color = 'red', label = '2010-2019 regression')
    ax[idx].plot(X,pred_2023,color = 'green', label = '2023 regression')

    ax[idx].legend()
    ax[idx].set_xlabel('First Weekend Audience')
    ax[idx].set_ylabel('Total Audience')

plt.tight_layout()
plt.savefig('figures/regression_plot_2023.png')

'''------------------------------------------------------------'''
'''Predict 2020 Jan'''
'''Commercial Foreign only using RF only'''

X, y = get_X_and_y(f'2020_Jan/commercial foreign.csv')
movie_name = X['movie_name']

X.drop(columns=['movie_code', 'movie_name'], inplace=True)

X.insert(loc=9,column = 'genre_공연',value = 0)
X.insert(loc=11,column = 'genre_기타',value = 0)
X.insert(loc=14,column = 'genre_멜로/로맨스',value = 0)
X.insert(loc=16,column = 'genre_미스터리',value = 0)
X.insert(loc=18,column = 'genre_사극',value = 0)
X.insert(loc=19,column = 'genre_서부극(웨스턴)',value = 0)
X.insert(loc=20,column = 'genre_성인물(에로)',value = 0)
X.insert(loc=25,column = 'genre_전쟁',value = 0)
X.insert(loc=28,column = 'distributor_CJ ENM',value = 0)
X.insert(loc=31,column = 'distributor_NEW',value = 0)
X.insert(loc=34,column = 'distributor_Universal',value = 0)
X.insert(loc=35,column = 'distributor_Warner',value = 0)
X.insert(loc=38,column = 'prod_stat_기타',value = 0)
X.insert(loc=39,column = 'type_etc',value = 0)

model = joblib.load(f'models/2010 - 2019/commercial foreign_rf_model.pkl')

df = pd.DataFrame()
df['movie_name'] = movie_name
df['Predicted Audience'] = model.predict(X)
df['Actual Audience'] = y

df.to_csv('data/predicted_audience_CF_2020_Jan.csv')

'''------------------------------------------------------------'''
'''Predict 2020 Feb'''
'''Commercial Foreign only using RF only'''

X, y = get_X_and_y(f'2020_Feb/commercial foreign.csv')
movie_name = X['movie_name']

X.drop(columns=['movie_code', 'movie_name'], inplace=True)
X.insert(loc=7,column = 'genre_SF',value = 0)
X.insert(loc=9,column = 'genre_공연',value = 0)
X.insert(loc=11,column = 'genre_기타',value = 0)
X.insert(loc=15,column = 'genre_뮤지컬',value = 0)
X.insert(loc=16,column = 'genre_미스터리',value = 0)
X.insert(loc=18,column = 'genre_사극',value = 0)
X.insert(loc=19,column = 'genre_서부극(웨스턴)',value = 0)
X.insert(loc=20,column = 'genre_성인물(에로)',value = 0)
X.insert(loc=28,column = 'distributor_CJ ENM',value = 0)
X.insert(loc=30,column = 'distributor_Lotte',value = 0)
X.insert(loc=31,column = 'distributor_NEW',value = 0)
X.insert(loc=33,column = 'distributor_ShowBox',value = 0)
X.insert(loc=38,column = 'prod_stat_기타',value = 0)
X.insert(loc=39,column = 'type_etc',value = 0)

model = joblib.load(f'models/2010 - 2019/commercial foreign_rf_model.pkl')

df = pd.DataFrame()
df['movie_name'] = movie_name
df['Predicted Audience'] = model.predict(X)
df['Actual Audience'] = y

df.to_csv('data/predicted_audience_CF_2020_Feb.csv')

'''------------------------------------------------------------'''
'''Predict 2020 Feb'''
'''Commercial Foreign only using RF only'''

X, y = get_X_and_y(f'2020_Feb/commercial korean.csv')
movie_name = X['movie_name']

X.drop(columns=['movie_code', 'movie_name'], inplace=True)


X.insert(loc=7,column = 'genre_SF',value = 0)
X.insert(loc=9,column = 'genre_공연',value = 0)
X.insert(loc=11,column = 'genre_기타',value = 0)
X.insert(loc=15,column = 'genre_뮤지컬',value = 0)
X.insert(loc=16,column = 'genre_미스터리',value = 0)
X.insert(loc=18,column = 'genre_사극',value = 0)
X.insert(loc=19,column = 'genre_서부극(웨스턴)',value = 0)
X.insert(loc=20,column = 'genre_성인물(에로)',value = 0)
X.insert(loc=28,column = 'distributor_CJ ENM',value = 0)
X.insert(loc=30,column = 'distributor_Lotte',value = 0)
X.insert(loc=31,column = 'distributor_NEW',value = 0)
X.insert(loc=33,column = 'distributor_ShowBox',value = 0)
X.insert(loc=38,column = 'prod_stat_기타',value = 0)
X.insert(loc=39,column = 'type_etc',value = 0)

model = joblib.load(f'models/2010 - 2019/commercial korean_rf_model.pkl')

df = pd.DataFrame()
df['movie_name'] = movie_name
df['Predicted Audience'] = model.predict(X)
df['Actual Audience'] = y

df.to_csv('data/predicted_audience_CK_2020_Feb.csv')

'''-------------------------------------------------'''
'''Clustering'''
'''Audience Total Type'''
'''Commercial Korean only'''
'''2010 - 2019'''

X, y = get_X_and_y(f'2010 - 2019/commercial korean.csv')
Xy = pd.DataFrame(X['AudienceCount'])
Xy['AudienceAcc'] = y

clustering = KMeans(n_clusters = 2, random_state = 42)
clustering.fit(Xy)

Xy['AudienceType'] = clustering.labels_

plt.figure()
plt.scatter(Xy['AudienceCount'],Xy['AudienceAcc'],c = Xy['AudienceType'])
plt.title('Audience - K-Means')
plt.tight_layout()
plt.savefig('figures/KMeans.png')

'''------------------------------------------------------------'''
'''Audience Total Type'''
'''Commercial Korean only'''
'''2010 - 2019'''

#get above/below regression line
X, y = get_X_and_y(f'2010 - 2019/commercial korean.csv')
X.drop(columns=['movie_code', 'movie_name'], inplace=True)

model = joblib.load(f'models/2010 - 2019/commercial korean_rf_model.pkl')
X_pred = model.predict(X)
X['Above_regression_line'] = (y>=X_pred)


# get first weekend below 200000 & acc above 2000000
X['AudienceAcc'] = y

def categorize_audience(count, acc):
    if count<200000 and acc>1000000:
        return True
    else:
        return False
X['AudienceType'] = X.apply(lambda row: categorize_audience(row['AudienceCount'],row['AudienceAcc']), axis = 1)


# get percentage of above regression line belonging to AudienceType = True

X['AboveReg_and_PatternB'] = X.apply(lambda row : (row['AudienceType']==True and row['Above_regression_line']== True), axis = 1)
X_above_reg = X.loc[X['Above_regression_line'] == True]

value_count = X_above_reg['AboveReg_and_PatternB'].value_counts(normalize=True)
print(f'Percentage of Pattern B among Above Regression Line : {value_count}')

X.to_csv('data/x.csv')

X_1 = X.loc[X['AudienceType']==False]
X_2 = X.loc[X['AudienceType']==True]


count1, count2 = len(X_1), len(X_2)

X_1_mean = X_1.mean()
X_2_mean = X_2.mean()

df = pd.concat((X_1_mean,X_2_mean),axis = 1)
df = df.transpose()
df['Count'] = [count1, count2]

df.to_csv('data/audience_type_mean_2010-2019.csv')
'''------------------------------------------------------------'''
# 패턴 A와 패턴 B 독립표본 t 검정
features_to_test = ['investor_count','company_count','star_ratings','AudienceCount', 'AudienceAcc',
                    'genre_드라마','genre_범죄', 'genre_사극', 'genre_액션',
                    'distributor_etc',
                    'rating_12','rating_15','rating_18','rating_all']
results = []

for feature in features_to_test:
    t_stat, p_value = stats.ttest_ind(X_1[feature],X_2[feature])
    mean_a = X_1[feature].mean()
    mean_b = X_2[feature].mean()
    results.append({
        'Feature' : feature,
        'T-stat' : t_stat,
        'P-value' : p_value,
        'Mean A' : mean_a,
        'Mean B' : mean_b
    })
results_df = pd.DataFrame(results)
results_df = results_df.transpose()
results_df.to_csv('data/pattern_a_b_tstat.csv')


'''------------------------------------------------------------'''
'''Audience Total Type'''
'''Commercial Korean only'''
'''2023'''


X, y = get_X_and_y(f'2023/commercial korean.csv')
X['AudienceAcc'] = y
X.drop(columns=['movie_code', 'movie_name'], inplace=True)

X['AudienceType'] = X.apply(lambda row: categorize_audience(row['AudienceCount'],row['AudienceAcc']), axis = 1)

X_1 = X.loc[X['AudienceType']==False]
X_2 = X.loc[X['AudienceType']==True]
count1, count2 = len(X_1), len(X_2)

X_1_mean = X_1.mean()
X_2_mean = X_2.mean()

df = pd.concat((X_1_mean,X_2_mean),axis = 1)
df = df.transpose()
df['Count'] = [count1, count2]

df.to_csv('data/audience_type_mean_2023.csv')

'''-------------------------------------------------'''
'''Pearson 상관계수 for first weekend audience'''
movie_data_list.insert(0,'movie_data.csv')

pearson = pd.DataFrame()
for filename in movie_data_list:
    X, y = get_X_and_y(f'2010 - 2019/{filename}')
    X.drop(columns=['movie_code', 'movie_name'], inplace=True)
    correlation, p_value = stats.pearsonr(X['AudienceCount'],y)
    pearson[f'{filename.split('.')[0]}'] = [correlation, p_value]
pearson.index = ['correlation','p-value']
pearson.to_csv('data/pearson.csv')

'''-------------------------------------------------'''
'''모델 평가'''

rmse_df = pd.DataFrame()
mae_df = pd.DataFrame()
r2_df = pd.DataFrame()
for filename in (movie_data_list):
    rmse_list = []
    mae_list = []
    r2_list = []
    X, y = get_X_and_y(f'2010 - 2019/{filename}')
    X.drop(columns=['movie_code', 'movie_name'], inplace=True)

    for modelname in model_list:
        model = joblib.load(f'models/2010 - 2019/{filename.split('.')[0]}_{modelname}')
        if modelname == 'lin_model_1.pkl':
            pred = model.predict(pd.DataFrame(X['AudienceCount']))
        else:
            pred = model.predict(X)
        rmse, mae, r2 = evaluate_model(y, pred)
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
    rmse_df[filename] = rmse_list
    mae_df[filename] = mae_list
    r2_df[filename] = r2_list
rmse_df.index = model_list
mae_df.index = model_list
r2_df.index = model_list

rmse_df.to_csv('data/rmse_df.csv')
mae_df.to_csv('data/mae_df.csv')
r2_df.to_csv('data/r2_df.csv')


'''-------------------------------------------------'''
'''모델 파라미터'''

params_df = pd.DataFrame()
for filename in ['movie_data.csv','commercial foreign.csv']:
    X, y = get_X_and_y(f'2010 - 2019/{filename}')
    X.drop(columns=['movie_code', 'movie_name'], inplace=True)
    col = []
    for modelname in model_list[2:]:
        model = joblib.load(f'models/2010 - 2019/{filename.split('.')[0]}_{modelname}')
        model_params = model.get_params()
        col.append(model_params)
    params_df[filename] = col
params_df.to_csv('data/model_params.csv')

'''-------------------------------------------------'''

'''모델 평가'''

rmse_df = pd.DataFrame()
mae_df = pd.DataFrame()
r2_df = pd.DataFrame()
for filename in (movie_data_list):
    rmse_list = []
    mae_list = []
    r2_list = []
    X, y = get_X_and_y(f'2023/{filename}')
    X.drop(columns=['movie_code', 'movie_name'], inplace=True)
    X.insert(loc=20, column='genre_성인물(에로)', value=0)

    for modelname in model_list:
        model = joblib.load(f'models/2010 - 2019/{filename.split('.')[0]}_{modelname}')
        if modelname == 'lin_model_1.pkl':
            pred = model.predict(pd.DataFrame(X['AudienceCount']))
        else:
            pred = model.predict(X)
        rmse, mae, r2 = evaluate_model(y, pred)
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
    rmse_df[filename] = rmse_list
    mae_df[filename] = mae_list
    r2_df[filename] = r2_list
rmse_df.index = model_list
mae_df.index = model_list
r2_df.index = model_list

rmse_df.to_csv('data/rmse_df_2023.csv')
mae_df.to_csv('data/mae_df_2023.csv')
r2_df.to_csv('data/r2_df_2023.csv')