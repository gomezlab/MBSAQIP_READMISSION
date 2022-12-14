import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from skopt import BayesSearchCV

# %%
#Open csv file
data = pd.read_feather('../../data_ugi/data10.feather')



# %%
#Split into training and test data
y = data['READ30']
X = data.drop(['READ30'], axis=1)

clf_xgb = XGBClassifier(tree_method='gpu_hist', use_label_encoder=False)

param_dist = {'n_estimators': [20, 50, 100, 200],
              'learning_rate': [0.03, 0.05, 0.075, 0.1, 0.3, 0.5],
              'subsample': [0.4, 0.6, 1.0],
              'max_depth': [6, 8, 12, 20],
              'colsample_bytree': [0.6, 0.8, 1.0],
              'min_child_weight': [2, 4, 6]
             }



clf = BayesSearchCV(clf_xgb, 
                         param_dist,
                         cv = 5,  
                         n_iter = 100, 
                         scoring = 'roc_auc', 
                         error_score = 0, 
                         verbose = 0, 
                         n_jobs = -1)
clf.fit(X, y)
results = pd.DataFrame(clf.cv_results_)
results.sort_values(by='rank_test_score').to_csv('../../results_ugi/results_xgb3_data10.csv')