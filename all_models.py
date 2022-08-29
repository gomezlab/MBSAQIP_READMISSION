# %%
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score
from tensorflow import keras
from math import sqrt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import csv

# %%
pd.options.display.max_rows = 20
pd.options.display.max_columns = 200

def roc_auc_ci(y_true, y_score, positive=1):
    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return (lower, AUC, upper)

def roc_prc_ci(y_true, y_score, positive=1):
    AUC = average_precision_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return (lower, AUC, upper)

# %%
train = pd.read_feather('../../data_ugi/data10.feather')
test = pd.read_feather('../../data_ugi/test10.feather')

# %%
train.head()

# %%
test.head()

# %% [markdown]
# train.drop(['index'], axis=1, inplace=True)
# test.drop(['index'], axis=1, inplace=True)

# %%
drop = ['READ30']
y = train['READ30']
X = train.drop(drop, axis=1)
y_test = test['READ30']
X_test = test.drop(drop, axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

# %%
y.value_counts()

# %%
y.value_counts(normalize=True)

# %%
y_test.value_counts()

# %%
X.head()

# %%
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2)

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)

# %%
rf = RandomForestClassifier(n_estimators=500, max_depth=20, max_features='auto', min_samples_leaf=8, min_samples_split=2, bootstrap=True, n_jobs=-1, random_state=0)
xgb = XGBClassifier(n_estimators=100, max_depth=6, colsample_bytree=0.6, learning_rate=0.03, min_child_weight=6, subsample=0.4, n_jobs=-1, random_state=0)
lr = LogisticRegression(penalty='none', random_state=0)

input_shape = X.shape[1:]
def build_model(n_hidden=4, n_neurons=500, dropout=0.2, learning_rate=3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(loss=keras.losses.BinaryCrossentropy(), metrics=['AUC'], optimizer=optimizer)
    return model
nn = build_model()


# %%
#calculate 95% confidence interval for auroc and auprc for each model (def = define function)
def auroc_ci(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    mean = roc_auc
    std = sqrt(roc_auc * (1.0 - roc_auc) / len(y_true))
    low  = mean - std
    high = mean + std
    return low, mean, high
#calculate auprc 95% ci for each model
def auprc_ci(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    mean = pr_auc
    std = sqrt(pr_auc * (1.0 - pr_auc) / len(y_true))
    low  = mean - std
    high = mean + std
    return low, mean, high
def fit_predict(model, X_train, y_train, X_test):
    model.fit(X, y)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return y_pred_proba


# %%
rf_y_pred = fit_predict(rf, X, y, X_test)
xgb_y_pred = fit_predict(xgb, X, y, X_test)
lr_y_pred = fit_predict(lr, X, y, X_test)


# %%
history = nn.fit(X_train, y_train,
                batch_size=512, epochs=500,
                validation_data=(X_valid, y_valid),
                callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)])


# %%
nn_y_pred = nn.predict(X_test)
rf_confidence = auroc_ci(y_test, rf_y_pred)
xgb_confidence = auroc_ci(y_test, xgb_y_pred)
lr_confidence = auroc_ci(y_test, lr_y_pred)
nn_confidence = auroc_ci(y_test, nn_y_pred)
print('Random Forest AUROC:', rf_confidence, 'AUROC CI:', rf_confidence)
print('XGBoost AUROC:', xgb_confidence, 'AUROC CI:', xgb_confidence)
print('Logistic Regression AUROC:', lr_confidence, 'AUROC CI:', lr_confidence)
print('Neural Network AUROC:', nn_confidence, 'AUROC CI:', nn_confidence)


# %%
#create labels for roc curves
rf_label = 'RF: ' + str(round(rf_confidence[1], 3)) + ' (' + str(round(rf_confidence[0], 3)) + ' - ' + str(round(rf_confidence[2], 3)) + ')'
xgb_label = 'XGB: ' + str(round(xgb_confidence[1], 3)) + ' (' + str(round(xgb_confidence[0], 3)) + ' - ' + str(round(xgb_confidence[2], 3)) + ')'
nn_label = 'NN: ' + str(round(nn_confidence[1], 3)) + ' (' + str(round(nn_confidence[0], 3)) + ' - ' + str(round(nn_confidence[2], 3)) + ')'
lr_label = 'LR: ' + str(round(lr_confidence[1], 3)) + ' (' + str(round(lr_confidence[0], 3)) + ' - ' + str(round(lr_confidence[2], 3)) + ')'
#calculate tpr and fpr for each model
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_y_pred)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_y_pred)
nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_y_pred)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_y_pred)


# %%
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
#plot the ROC curves for each model
plt.figure(figsize=(10,10))
plt.plot(lr_fpr, lr_tpr, color='red', label=lr_label)
plt.plot(rf_fpr, rf_tpr, color='deepskyblue', label=rf_label)
plt.plot(xgb_fpr, xgb_tpr, color='steelblue', label=xgb_label)
plt.plot(nn_fpr, nn_tpr, color='dodgerblue', label=nn_label)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.legend(loc="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([0.0, 1.0])
plt.savefig('../../results_ugi/roc_nopostop10.png')


# %%
rf_auprc_ci = auprc_ci(y_test, rf_y_pred)
xgb_auprc_ci = auprc_ci(y_test, xgb_y_pred)
lr_auprc_ci = auprc_ci(y_test, lr_y_pred)
nn_auprc_ci = auprc_ci(y_test, nn_y_pred)
#calculate precision and recall for each model
rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_y_pred)
xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, xgb_y_pred)
nn_precision, nn_recall, _ = precision_recall_curve(y_test, nn_y_pred)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_y_pred)
#create labels for precision recall curves
rf_prc_label = 'RF: ' + str(round(rf_auprc_ci[1], 3)) + ' (' + str(round(rf_auprc_ci[0], 3)) + ' - ' + str(round(rf_auprc_ci[2], 3)) + ')'
xgb_prc_label = 'XGB: ' + str(round(xgb_auprc_ci[1], 3)) + ' (' + str(round(xgb_auprc_ci[0], 3)) + ' - ' + str(round(xgb_auprc_ci[2], 3)) + ')'
nn_prc_label = 'NN: ' + str(round(nn_auprc_ci[1], 3)) + ' (' + str(round(nn_auprc_ci[0], 3)) + ' - ' + str(round(nn_auprc_ci[2], 3)) + ')'
lr_prc_label = 'LR: ' + str(round(lr_auprc_ci[1], 3)) + ' (' + str(round(lr_auprc_ci[0], 3)) + ' - ' + str(round(lr_auprc_ci[2], 3)) + ')'
#plot the precision recall curves for each model
matplotlib.rcParams.update({'font.size': 16})
#plot the ROC curves for each model
plt.figure(figsize=(10,10))
plt.plot(lr_recall, lr_precision, color='red', label=lr_prc_label)
plt.plot(rf_recall, rf_precision, color='deepskyblue', label=rf_prc_label)
plt.plot(xgb_recall, xgb_precision, color='steelblue', label=xgb_prc_label)
plt.plot(nn_recall, nn_precision, color='dodgerblue', label=nn_prc_label)
plt.legend(loc="upper right")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0.0, 1.0])
plt.savefig('../../results_ugi/prc_nopostop10.png')


# %%
xgb.fit(X, y)
fi = xgb.feature_importances_
fi_sorted = np.argsort(fi)
fi_sorted = fi_sorted[::-1]
fi_sorted = fi_sorted[:15]


# %%
#plot the top 15 feature on a horizontal bar chart, with highest on the top
plt.figure(figsize=(10,10))
plt.barh(np.arange(15), fi[fi_sorted], color='steelblue')
plt.yticks(np.arange(15), X.columns[fi_sorted])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.savefig('../../results_ugi/xgb_fi_nopostop10.png', bbox_inches='tight')


# %%
from sklearn.metrics import recall_score
from imblearn.metrics import specificity_score
import numpy as np
thresh = np.arange(0, 1, 0.00001)
#create a dataframe to store the sensitivity and specificity at each threshold for each model
lr_senspec = pd.DataFrame(columns=['thresh', 'sens','spec'])
xgb_senspec = pd.DataFrame(columns=['thresh', 'sens','spec'])
lr_sens = {}
lr_spec = {}
xgb_sens = {}
xgb_spec = {}
for t in thresh:
    lr_sens[t] = recall_score(y_test, lr_y_pred > t)
    lr_spec[t] = specificity_score(y_test, lr_y_pred > t)
    xgb_sens[t] = recall_score(y_test, xgb_y_pred > t)
    xgb_spec[t] = specificity_score(y_test, xgb_y_pred > t)
#add each dictionary to the dataframe
lr_senspec['thresh'] = lr_sens.keys()
lr_senspec['sens'] = lr_sens.values()
lr_senspec['spec'] = lr_spec.values()
xgb_senspec['thresh'] = xgb_sens.keys()
xgb_senspec['sens'] = xgb_sens.values()
xgb_senspec['spec'] = xgb_spec.values()
#plot the sensitivity and specificity
plt.plot(lr_senspec['thresh'], lr_senspec['sens'], label='LR')
plt.plot(xgb_senspec['thresh'], xgb_senspec['sens'], label='xgb')
plt.plot(lr_senspec['thresh'], lr_senspec['spec'], label='LR')
plt.plot(xgb_senspec['thresh'], xgb_senspec['spec'], label='xgb')
plt.legend()
plt.xlabel('Threshold')
plt.ylabel('Sensitivity/Specificity')
plt.title('Sensitivity/Specificity vs Threshold')
plt.show()

# %%
xgb_senatspec = {}
lr_senatspec = {}
#find the value for xgb sensitivity where specificity is close to 90%
xgb_senatspec[90] = float(str(xgb_senspec['sens'].loc[round(xgb_senspec['spec'],1) == 0.900]).split()[1])
lr_senatspec[90] = float(str(lr_senspec['sens'].loc[round(lr_senspec['spec'],1) == 0.900]).split()[1])

xgb_senatspec[70] = float(str(xgb_senspec['sens'].loc[round(xgb_senspec['spec'],1) == 0.700]).split()[1])
lr_senatspec[70] = float(str(lr_senspec['sens'].loc[round(lr_senspec['spec'],1) == 0.700]).split()[1])
xgb_senatspec[50] = float(str(xgb_senspec['sens'].loc[round(xgb_senspec['spec'],1) == 0.500]).split()[1])
lr_senatspec[50] = float(str(lr_senspec['sens'].loc[round(lr_senspec['spec'],1) == 0.500]).split()[1])
xgb_senatspec[30] = float(str(xgb_senspec['sens'].loc[round(xgb_senspec['spec'],1) == 0.300]).split()[1])
lr_senatspec[30] = float(str(lr_senspec['sens'].loc[round(lr_senspec['spec'],1) == 0.300]).split()[1])


# %%
xgb_senatspec[10] = float(str(xgb_senspec['sens'].loc[round(xgb_senspec['spec'],1) == 0.100]).split()[1])
lr_senatspec[10] = float(str(lr_senspec['sens'].loc[round(lr_senspec['spec'],1) == 0.100]).split()[1])

# %%
xgb_senspec['spec'].hist()

# %%
#write xgb_senatspec to a csv file
xgb_senatspec_df = pd.DataFrame.from_dict(xgb_senatspec, orient='index')
xgb_senatspec_df.columns = ['sensitivity']
#rename index to 'specificity'
xgb_senatspec_df.index.name = 'specificity'
xgb_senatspec_df.to_csv('../../results_ugi/xgb_senspec_nopostop10.csv')
lr_senatspec_df = pd.DataFrame.from_dict(lr_senatspec, orient='index')
lr_senatspec_df.columns = ['sensitivity']
#rename index to 'specificity'
lr_senatspec_df.index.name = 'specificity'
lr_senatspec_df.to_csv('../../results_ugi/lr_senspec_nopostop10.csv')


