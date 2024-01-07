import torch
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import mean_absolute_error
import math
import pandas as pd
import joblib
import sys
from sklearn.svm import SVR
from sklearn.tree import ExtraTreeRegressor,DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
import os
from sklearn.model_selection import GridSearchCV


def train(model_name,inds_path, model_save_path, train_real_pred__save_path,test_real_pred_save_path,score_save_path,
      name,cation_smiles,anion_smiles,smiles,tem,y):

    R2Score_trains, R2Score_tests = [], []
    MSE_trains, MSE_tests = [], []
    RMSE_trains, RMSE_tests = [], []
    MAE_trains, MAE_tests = [], []

    for i in range(5):
        ML_model = []
        XGB_model = xgb.XGBRegressor(booster='gbtree',
                                     max_depth=5,
                                     learning_rate=0.1,
                                     n_estimators=200,
                                     eta=0.3,
                                     gamma=0,
                                     sampling_method='uniform')
        ML_model.append(XGB_model)


        DT_model = DecisionTreeRegressor(criterion="squared_error",
                                              splitter='best',
                                              max_depth=50,
                                              min_samples_split=2,
                                              min_samples_leaf=1,
                                              min_weight_fraction_leaf=0.0,
                                              max_features=2,
                                              random_state=0,
                                              max_leaf_nodes=None,
                                              min_impurity_decrease=0.0,
                                              ccp_alpha=0.0)
        ML_model.append(DT_model)

        RF_model = RandomForestRegressor(n_estimators=200,
                                         criterion='squared_error',
                                         max_depth=20,
                                         min_samples_split=2,
                                         min_samples_leaf=1,
                                         min_weight_fraction_leaf=0.0,
                                         max_features=5,
                                         max_leaf_nodes=None,
                                         min_impurity_decrease=0.0,
                                         bootstrap=True,
                                         oob_score=False,
                                         n_jobs=None,
                                         random_state=0,
                                         verbose=0,
                                         warm_start=False,
                                         ccp_alpha=0.0,
                                         max_samples=None)
        ML_model.append(RF_model)

        GBRT_model = GradientBoostingRegressor(loss='squared_error',
                                               learning_rate=0.1,
                                               n_estimators=200,
                                               subsample=1.0,
                                               criterion='friedman_mse',
                                               min_samples_split=2,
                                               min_samples_leaf=1,
                                               min_weight_fraction_leaf=0.0,
                                               max_depth=5,
                                               min_impurity_decrease=0.0,
                                               init=None,
                                               random_state=None,
                                               max_features=None,
                                               alpha=0.9,
                                               verbose=0,
                                               max_leaf_nodes=None,
                                               warm_start=False,
                                               validation_fraction=0.1,
                                               n_iter_no_change=None,
                                               tol=0.0001,
                                               ccp_alpha=0.0)
        ML_model.append(GBRT_model)

        param_grid = {
            'kernel':['rbf'],
            'C':[1000,2000,3000],
            'degree': [1,2,3],
            'tol':[0.1,0.01,0.001,0.0001,0.00001],
            'epsilon':[0.1,0.01,0.001,0.0001,0.00001],
            'cache_size':[10,20,50,100,200,300]
        }

        SVR_model = SVR()

        # SVR_model = SVR(C=300,
        #                 kernel='rbf',
        #                 degree=1,
        #                 gamma=0.0001,
        #                 shrinking=True,
        #                 tol=0.000001,
        #                 cache_size=200,
        #                 epsilon=0.00001,
        #                 verbose=False,
        #                 max_iter=-1)

        SVR_model = SVR(C=3000,
                        kernel='rbf',
                        degree=1,
                        tol=0.0001,
                        cache_size=10,
                        epsilon=0.001,
                       )

        ML_model.append(SVR_model)

        ML_name = ['XGB', 'DT', 'RF', 'GBRT', 'SVR']

        ML_dict = dict(zip(ML_name, ML_model))

        train_index = pd.read_csv(os.path.join(inds_path, r'train_ind_' + str(i) + '.csv'))
        test_index = pd.read_csv(os.path.join(inds_path, r'test_ind_' + str(i) + '.csv'))
        train_index = np.array(train_index['train_ind'])
        test_index = np.array(test_index['test_ind'])
        X_train, y_train, names_train, cation_smiles_train, anion_smiles_train, smiles_train, tem_train = X[train_index],y[train_index],name[train_index],cation_smiles[train_index],anion_smiles[train_index],smiles[train_index], tem[train_index]
        X_test, y_test, names_test, cation_smiles_test, anion_smiles_test, smiles_test, tem_test = X[test_index], y[test_index], name[test_index], cation_smiles[test_index], anion_smiles[test_index], smiles[test_index], tem[test_index]

        y_train = torch.tensor(y_train).to(torch.float32)
        model = ML_dict[model_name]
        model.fit(X_train, y_train)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        print(grid_search.best_params_)
        print(grid_search.best_score_)
        joblib.dump(model, model_save_path + str(i) + '.pkl')

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        R2Score_train = r2_score(y_train, y_train_pred)
        R2Score_trains.append(R2Score_train)
        R2Score_test = r2_score(y_test, y_test_pred)
        R2Score_tests.append(R2Score_test)

        MSE_train = mean_squared_error(y_train, y_train_pred)
        MSE_trains.append(MSE_train)
        MSE_test = mean_squared_error(y_test, y_test_pred)
        MSE_tests.append(MSE_test)

        RMSE_train = math.sqrt(MSE_train)
        RMSE_trains.append(RMSE_train)
        RMSE_test = math.sqrt(MSE_test)
        RMSE_tests.append(RMSE_test)

        MAE_train = mean_absolute_error(y_train, y_train_pred)
        MAE_trains.append(MAE_train)
        MAE_test = mean_absolute_error(y_test, y_test_pred)
        MAE_tests.append(MAE_test)

        dataframe = pd.DataFrame(
            {'names': names_train, 'cation_smiles_train': cation_smiles_train, 'anion_smiles_train': anion_smiles_train,
             'smiles': smiles_train, 'temperature': tem_train, 'y_train': y_train, 'y_train_pred': y_train_pred})
        dataframe.to_csv(train_real_pred__save_path + str(i) + '.csv', sep=',', index=False)
        dataframe = pd.DataFrame(
            {'names': names_test, 'cation_smiles_test': cation_smiles_test, 'anion_smiles_test': anion_smiles_test,
             'smiles': smiles_test, 'temperature': tem_test, 'y_test': y_test, 'y_test_pred': y_test_pred})
        dataframe.to_csv(test_real_pred_save_path + str(i) + '.csv', sep=',', index=False)

    train_Score = [str(round(np.mean(R2Score_trains), 4)) + '+-' + str(round(np.std(R2Score_trains), 4)),
                   str(round(np.mean(MSE_trains), 4)) + '+-' + str(round(np.std(MSE_trains), 4)),
                   str(round(np.mean(RMSE_trains), 4)) + '+-' + str(round(np.std(RMSE_trains), 4)),
                   str(round(np.mean(MAE_trains), 4)) + '+-' + str(round(np.std(MAE_trains), 4))]
    test_Score = [str(round(np.mean(R2Score_tests), 4)) + '+-' + str(round(np.std(R2Score_tests), 4)),
                  str(round(np.mean(MSE_tests), 4)) + '+-' + str(round(np.std(MSE_tests), 4)),
                  str(round(np.mean(RMSE_tests), 4)) + '+-' + str(round(np.std(RMSE_tests), 4)),
                  str(round(np.mean(MAE_tests), 4)) + '+-' + str(round(np.std(MAE_tests), 4))]

    print(R2Score_trains)
    print(R2Score_tests)

    df_Score = pd.DataFrame({'train_Score': train_Score, 'test_Score': test_Score},
                            index=['R2', 'MSE', 'RMSE', 'MAE'])
    print(df_Score)
    df_Score.to_csv(score_save_path )
    print(model_name+"finish!!!")


ML = ['SVR']
data_path = "../datasets/ionic_conductivity.csv"
Xs_Path = "../datasets/Xs.pt"
inds_path = '../datasets/indx_id/'

for i in ML:
    model_save_path = "../model/ML_model/"+i+"/model_"
    train_real_pred__save_path = "../result/"+i+"/train_real_pred/train_real_pred_"
    test_real_pred_save_path = "../result/"+i+"/test_real_pred/test_real_pred_"
    score_save_path = "../result/"+i+"/score.csv"

    df = pd.read_csv(data_path)
    name = df['Name'].values
    cation_smiles = df['Cation'].values
    anion_smiles = df['Anion'].values
    smiles = df['Smiles'].values
    tem = df['Temperature, K'].values
    y = df['ActualValue'].values

    X = torch.load(Xs_Path)
    X = torch.tensor(np.array(X)).to(torch.float32)
    y = np.array(y).squeeze()

    train(i,inds_path, model_save_path, train_real_pred__save_path, test_real_pred_save_path, score_save_path,
          name, cation_smiles, anion_smiles, smiles, tem, y)














