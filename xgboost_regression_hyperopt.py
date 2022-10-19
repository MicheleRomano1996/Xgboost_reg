# function for xgboort regressor
# scoring: rmse, mape

import pandas as pd
from math import *
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from xgboost import XGBRegressor, plot_importance

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope

import warnings
warnings.filterwarnings("ignore")



def calcError(y_test, y_pred):
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = sqrt(MSE)
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    print(f'Root Mean Sqarred Error: {round(RMSE,4)}')
    print(f'Mean Absolute Percentage Error %: {round(MAPE*100,4)}')
    return



def xgboost_regressor(X_train_initial,y_train_initial,X_test,y_test,score,space,n_evals):
    
    # -> baseline error
    mean_error = y_train_initial.mean()
    baseline_array = np.ones(len(y_train_initial))*mean_error
    print('-> Baseline error (mean of Y target):')
    calcError(baseline_array,y_train_initial.values)

    
    # -> split train validation 
    X_train, X_val, y_train, y_val = train_test_split(X_train_initial, y_train_initial, test_size = 0.20, random_state=42)

    
    # -> fitting initial xgb model and cv error
    xgb_model = XGBRegressor(n_estimators = 300,
                                 learning_rate=0.1,
                                 seed = 42)
    cv = StratifiedKFold(n_splits=10,shuffle=True,random_state = 42)
    scores_rmse = cross_val_score(xgb_model, 
                         X_train_initial, 
                         y_train_initial, 
                         scoring='neg_mean_squared_error', 
                         cv=cv, 
                         n_jobs=-1)
    scores_mape = cross_val_score(xgb_model, 
                         X_train_initial, 
                         y_train_initial, 
                         scoring='neg_mean_absolute_percentage_error', 
                         cv=cv, 
                         n_jobs=-1)
    print('\n\n -> Error on train data cv:')
    print(f'Root Mean Sqarred Error: {round(sqrt(abs(scores_rmse).mean()),4)}')
    print(f'Mean Absolute Percentage Error %: {round(abs(scores_mape).mean()*100,4)}\n')
    
    
    # -> fitting initial xgb model on validation data
    xgb_model.fit(X_train,y_train)
    y_pred_val = xgb_model.predict(X_val)
    print('\n\n -> Error initial xgb model on val data:')
    calcError(y_val,y_pred_val)
    
    
    
    # -> Bayesian optimization: hyperopt 
    print('\n\n -> Starting hyperparams tuning with hyperopt')
    
    # 1) Define the space over which hyperopt will search for optimal hyperparameters.
    # --> space
    
    # 2) define obj function
    def hyperparameter_tuning(space):
        model=XGBRegressor(**space,
                           eval_metric=score,
                           early_stopping_rounds=10)
        evaluation = [(X_train, y_train), (X_val, y_val)]
        model.fit(X_train, y_train,
                  eval_set=evaluation,
                  verbose=False)
        pred = model.predict(X_val)
        # define objective function
        if score == 'rmse':
            metric = mean_squared_error(y_val, pred, squared=False)
        elif score == 'mape':
            metric = mean_absolute_percentage_error(y_val, pred)
        return {'loss':metric, 'status': STATUS_OK, 'model': model}
    
    # 3) Run for n_evals trials
    trials = Trials()
    best = fmin(fn=hyperparameter_tuning,
                space=space,
                algo=tpe.suggest,
                max_evals=n_evals,
                trials=trials)
    print()
    print(f'\nbest params {score}:',best)
    
    
    # -> fit final model after hyperopt and display errors
    xgb_final_hyperopt = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
    y_pred_xgb_hyperopt = xgb_final_hyperopt.predict(X_val)
    print('\n\n -> Error on val data after tuning hyperopt:')
    calcError(y_val,y_pred_xgb_hyperopt)
    
    if score == 'rmse':
        initial_error_xgb_hyperopt = sqrt(mean_squared_error(y_val,y_pred_val))
        final_error_xgb_hyperopt = sqrt(mean_squared_error(y_val,y_pred_xgb_hyperopt))
    elif score == 'mape':
        initial_error_xgb_hyperopt = mean_absolute_percentage_error(y_val,y_pred_val)*100
        final_error_xgb_hyperopt = mean_absolute_percentage_error(y_val,y_pred_xgb_hyperopt)*100
    perc_gain = round(initial_error_xgb_hyperopt/final_error_xgb_hyperopt*100-100,2)
    
    print('\n\n -> Initial vs hyperopt error: ')
    print(f'initial error {score} val data: {round(initial_error_xgb_hyperopt,2)}')
    print(f'final error {score} val data: {round(final_error_xgb_hyperopt,2)}')
    print(f'% of error {score} gained after tuning hyperparams: {perc_gain} %')
    
    
    # -> fitting initial xgb model on test data
    y_pred_initial = xgb_model.predict(X_test)
    print('\n\n -> Error initial xgb model on test data:')
    calcError(y_test,y_pred_initial)
    
    rmse_initial = sqrt(mean_squared_error(y_test,y_pred_initial))
    mape_initial = mean_absolute_percentage_error(y_test,y_pred_initial)*100
    
    # -> error hyperopt xgb model on test data
    y_test_hyperopt = xgb_final_hyperopt.predict(X_test)
    print('\n\n -> Error hyperopt xgb model on test data:')
    calcError(y_test,y_test_hyperopt)
    
    rmse_hyperopt = sqrt(mean_squared_error(y_test,y_test_hyperopt))
    mape_hyperopt = mean_absolute_percentage_error(y_test,y_test_hyperopt)*100
    
    print('\n\n -> Final results:')
    print(f'% hyperopt/initial rmse: {round(rmse_hyperopt/rmse_initial*100-100,2)} %')
    print(f'% hyperopt/initial mape: {round(mape_hyperopt/mape_initial*100-100,2)} %')
    
    
    # -> fitting initial xgb model on test data
    xgb_model.fit(X_train_initial,y_train_initial)
    y_pred_initial_fit = xgb_model.predict(X_test)
    print('\n\n -> Error initial xgb model on test after fitting on entire train set:')
    calcError(y_test,y_pred_initial_fit)
    
    rmse_initial_fit = sqrt(mean_squared_error(y_test,y_pred_initial_fit))
    mape_initial_fit = mean_absolute_percentage_error(y_test,y_pred_initial_fit)*100
    
    # -> error hyperopt xgb model on test data
    xgb_final_hyperopt.fit(X_train_initial,
            y_train_initial,
            eval_set=[(X_train_initial,y_train_initial), (X_test,y_test)],
            verbose=False)
    y_test_hyperopt_fit = xgb_final_hyperopt.predict(X_test)
    print('\n\n -> Error hyperopt xgb model on test data after fitting on entire train set:')
    calcError(y_test,y_test_hyperopt_fit)
    
    rmse_hyperopt_fit = sqrt(mean_squared_error(y_test,y_test_hyperopt_fit))
    mape_hyperopt_fit = mean_absolute_percentage_error(y_test,y_test_hyperopt_fit)*100
    
    print('\n\n -> Final results aafter fitting on entire train set:')
    print(f'% hyperopt/initial rmse: {round(rmse_hyperopt_fit/rmse_initial_fit*100-100,2)} %')
    print(f'% hyperopt/initial mape: {round(mape_hyperopt_fit/mape_initial_fit*100-100,2)} %')
    
    # return:
    # - xgb_model: xgb initial model fit on entire X_train
    # - xgb_final_hyperopt: xgb after hyperopt tuning fit on entire y_train
    
    return xgb_model, xgb_final_hyperopt


