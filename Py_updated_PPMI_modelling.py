# updated script for sPD/LRRK2-PD classification
# 1. Preparation
import numpy as np
import pandas as pd
import os
df = pd.read_table("PPMI_Model_Classification_all.DEA.genes_BL.txt")
df.rename(columns = {'pheno.code':'phenocode'}, inplace = True)
## 1.1 Data overview: check missing values
missing_count = df.isnull().sum()
value_count = df.isnull().count()
missing_percentage = round(missing_count/value_count*100,2)
missing_df = pd.DataFrame({"count":missing_count, "percentage":missing_percentage})
print(missing_df)
df_tidy = df.dropna()

## 1.2 train-test split (ratio 2:1)
from sklearn.model_selection import train_test_split
x = df_tidy.drop(columns = ["phenocode"])
y = df_tidy.phenocode
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

## 1.2 Resampling: to balance sPD and LRRK2-PD cohort by computing observations for LRRK2-PD
## oversampling can only be performed afte train-test split since it will artificially balance the test set
from sklearn.utils import resample
df_train = pd.DataFrame(x_train)
df_train["phenocode"] = y_train
df_train_majority = df_train[df_train.phenocode==0]
df_train_minority = df_train[df_train.phenocode==1]
df_train.phenocode.value_counts()
df_train_minority_upsampled = resample(df_train_minority, replace = True, n_samples=299, random_state = 123)
df_train_resample = pd.concat([df_train_minority_upsampled, df_train_majority])
x_train = df_train_resample.drop(columns=["phenocode"])
y_train = df_train_resample.phenocode

# 2.Model selection
#?? lasso regression with 10 fold cross validation
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
x_train_scaled = StandardScaler().fit_transform(x_train)
x_train_scaled = pd.DataFrame(x_train_scaled, index = x_train.index, columns = x_train.columns)
model = LassoCV(cv = 10, random_state = 0, max_iter = 100000)
model.fit(x_train_scaled, y_train)
model.alpha_ 
lasso_best = Lasso(alpha=model.alpha_)
lasso_best.fit(x_train_scaled, y_train)
print(list(zip(lasso_best.coef_, x_train)))

# all other classifiers

# all other classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
model_pipeline = []
model_pipeline.append(LogisticRegression(solver = "liblinear"))
model_pipeline.append(RandomForestClassifier())
model_pipeline.append(SVC(probability=True))
# 10 fold validation: AUROC, AUPRC, calibration
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import RocCurveDisplay, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
list_AP = []
list_auc = []
list_accuracy = []
list_recall = []
list_f1 = []
for model in model_pipeline:
        pipeline = make_pipeline(StandardScaler(), model)
        print(model)
        auc = cross_val_score(pipeline, X=x_train, y = y_train, cv=10, scoring = "roc_auc")# cv = 10, k
        f1 = cross_val_score(pipeline, X=x_train, y = y_train, cv = 10, scoring = "f1")
        accuracy = cross_val_score(pipeline, X=x_train, y = y_train, cv = 10, scoring = "accuracy")
        recall = cross_val_score(pipeline, X=x_train, y = y_train, cv = 10, scoring = "recall")
        AP = cross_val_score(pipeline,X=x_train, y = y_train, cv = 10, scoring = "average_precision")
        print(auc, f1, accuracy, recall, AP)
        list_AP.append(AP)
        list_auc.append(auc)
        list_accuracy.append(accuracy)
        list_recall.append(recall)
        list_f1.append(f1)

list_AP = pd.DataFrame(list_AP, index = ['LR', 'RF', 'SVC']).transpose()
list_auc = pd.DataFrame(list_auc, index = ['LR', 'RF', 'SVC']).transpose()
list_accuracy = pd.DataFrame(list_accuracy, index = ['LR', 'RF', 'SVC']).transpose()
list_recall = pd.DataFrame(list_recall, index = ['LR', 'RF', 'SVC']).transpose()
list_f1 = pd.DataFrame(list_f1, index = ['LR', 'RF', 'SVC']).transpose()

# Kruskal-Wallis Test & box plot
import seaborn as sns
import matplotlib.pyplot as plt
plt.subplot(121)
sns.boxplot(data=list_AP).set(title = "Average Precision", )
plt.subplot(122)
sns.boxplot(data=list_auc).set(title = "AUROC", )
plt.show()
plt.clf()             
from scipy import stats                
stats.kruskal(list_AP.LogisticRegression, list_AP.RandomForest, list_AP.SVC)                
stats.kruskal(list_auc.LogisticRegression, list_auc.RandomForest, list_auc.SVC)                  
## NOTE: logistic regression is the best model   


#ROC and PR plot
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import metrics
plt.rcParams.update({'font.size': 20})
model_list = ["LogisticRegression", "RandomForestClassifier", "SVC"]
i = 0
for model in model_pipeline:
        model.fit(x_train, y_train)
        y_pred = model.predict_proba(x_train)
        # ROC
        fpr, tpr, _thresholds = metrics.roc_curve(y_train, y_pred)
        label = model_list[i]+": AUC=" +  str(round(metrics.auc(fpr,tpr),2))
        plt.plot(fpr, tpr, label = label)
        i = i+1

plt.title("AUROC")
plt.legend(loc = "lower right")
plt.show()
plt.clf()

i = 0
for model in model_pipeline:
        y_pred_probs = model.predict_proba(x_train)
        y_pred_probs = y_pred_probs[:,1]
        precision, recall, thresholds = precision_recall_curve(y_train, y_pred_probs)
        label = model_list[i]+": AP=" +  str(round(metrics.auc(recall, precision),2))
        plt.plot(recall, precision, label = label)
        i = i+1

plt.legend(loc = "lower center")
plt.title("AUPRC")
plt.show()
plt.clf()  

# Calibration
from sklearn.calibration import calibration_curve
from matplotlib import pyplot
# plot perfectly calibrated

i = 0
for model in model_pipeline:
        y_pred_probs = model.predict_proba(x_train)
        y_pred_probs = y_pred_probs[:,1]
        fop, mpv = calibration_curve(y_train, y_pred_probs, n_bins=10)
        label = model_list[i]
        pyplot.plot(mpv, fop, marker='.')
        i = i+1

pyplot.title("Calibration Curve")
pyplot.legend(model_list, loc="upper left")
pyplot.plot([0, 1], [0, 1], linestyle='--', color = "black")             
pyplot.show()
plt.clf()

# 3.feature selection: permutation factor importance
from sklearn.inspection import permutation_importance
from sklearn.mtrixs import mean_squared_error
from sklearn.feature_selection import SelectFromModel

best_model = LogisticRegression(solver = "liblinear")
ss = StandardScaler()
x_train_scaled = ss.fit_transform(x_train)
x_train_scaled = pd.DataFrame(x_train_scaled, index = x_train.index, columns = x_train.columns)
best_model.fit(x_train_scaled,y_train)
# permutation for AP
plt.rcParams.update({'font.size': 8})
premu_lr_train_AP = permutation_importance(best_model, x_train_scaled, y_train, scoring="average_precision")
importance_AP_mean = premu_lr_train_AP.importances_mean[premu_lr_train_AP.importances_mean>0.05]
AP_genes = x_train.columns[premu_lr_train_AP.importances_mean>0]
sorted_idx = importance_AP_mean.argsort()
plt.tight_layout()
plt.barh(range(len(sorted_idx)), importance_AP_mean[sorted_idx], align = "center")
plt.yticks(range(len(sorted_idx)), np.array(AP_genes)[sorted_idx])
plt.title("Permutation Importance (Average Precision)")
plt.show()
plt.clf()
importance_AP_sd = premu_lr_train_AP.importances_std[premu_lr_train_AP.importances_mean>0]
var = AP_genes
per_res_AP_trimmed = pd.DataFrame({'interactor':var, 'importance_AP_mean':importance_AP_mean, 'importance_AP_std':importance_AP_sd}).sort_values(by=['importance_AP_mean'], ascending=False)
per_res_AP_trimmed.to_csv("PPMI_modelling_permutation_AP_BL.txt", sep = "\t")
# permutation for auc
premu_lr_train_auc = permutation_importance(best_model, x_train_scaled, y_train, scoring="roc_auc")
importance_auc_mean = premu_lr_train_auc.importances_mean[premu_lr_train_auc.importances_mean>0]
auc_genes = x_train.columns[premu_lr_train_auc.importances_mean>0]
sorted_idx = importance_auc_mean.argsort()
plt.tight_layout()
plt.barh(range(len(sorted_idx)), importance_auc_mean[sorted_idx], align = "center", color = 'orange')
plt.yticks(range(len(sorted_idx)), np.array(auc_genes)[sorted_idx])
plt.title("Permutation Importance (AUROC)")
plt.show()
plt.clf()
var=auc_genes
importance_auc_sd = premu_lr_train_auc.importances_std[premu_lr_train_auc.importances_mean>0]
per_res_auc_trimmed = pd.DataFrame({'interactor':var, 'importance_auc_mean':importance_auc_mean, 'importance_auc_std':importance_auc_sd}).sort_values(by=['importance_auc_mean'], ascending=False)
per_res_auc.to_csv("PPMI_modelling_permutation_auc_BL.txt", sep = "\t")
# feature selection: combination of predictors for highest AP and auc [note:I just skipped this and kept all predictors]
i = 1
mean_AP = []
sd_AP = []
mean_auc = []
sd_auc = []
for i in range(5,50):
        var_index = range(0,i)
        # AP selection
        var_list1 = np.array(pre_res_AP_trimmed['interactor'])[var_index]
        temp_X1 = x_train[var_list1]
        best_model.fit(temp_X1, y_train)
        pipeline = make_pipeline(StandardScaler(), best_model)
        AP = cross_val_score(pipeline, X=temp_X1, y = y_train, cv=10, scoring = "average_precision")
        mean_AP.append(AP.mean())
        sd_AP.append(AP.std())
        # auc selection
        var_list2 = np.array(per_res_auc_trimmed['interactor'])[var_index]
        temp_X2 = x_train[var_list2]
        best_model.fit(temp_X2, y_train)
        pipeline = make_pipeline(StandardScaler(), best_model)
        auc = cross_val_score(pipeline, X=temp_X2, y = y_train, cv=10, scoring = "roc_auc")
        mean_auc.append(auc.mean())
        sd_auc.append(auc.std())

# plot 
plt.subplot(121)
sns.scatterplot(x = range(5,50), y = mean_AP).set(title = "Average Precision")
plt.subplot(122)
sns.scatterplot(x = range(5,50), y = mean_auc).set(title = "AUROC")
plt.show()
plt.clf()   

var_list_final = per_res_AP_trimmed['interactor'].append(per_res_auc_trimmed['interactor'])
var_list_final.drop_duplicates(inplace = True)

#  3. model testing and evaluation
plt.rcParams.update({'font.size': 10})
plt.tight_layout()
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
x_test_trimmed = x_test
x_test_scaled = ss.transform(x_test)
x_test_scaled = pd.DataFrame(x_test_scaled, index = x_test.index, columns = x_test.columns)
# best_model.fit(x_test_trimmed, y_test)
y_pred = best_model.predict_proba(x_test_scaled)[:,1]
res_list = pd.DataFrame()
res_list["y_pred"] = y_pred
col_name = res_list.columns
acc_test = metrics.accuracy_score(y_test,y_pred)
fpr, tpr, _thresholds = metrics.roc_curve(y_test, y_pred)
auc  = round(metrics.auc(fpr,tpr),2)
cm=confusion_matrix(y_test,y_pred)
label = "Logistic Regression"+": AUC=" +  str(round(metrics.auc(fpr,tpr),2))
plt.legend(loc="lower right")
plt.title("AUROC")
plt.plot(fpr, tpr, label = label)
plt.show()
plt.clf()

cm
plt.rcParams.update({'font.size': 10})
cm_plot = sns.heatmap(cm, annot = True, cmap = 'Blues_r')
cm_plot.set_xlabel("Predicted values")
cm_plot.set_ylabel("Actual values")
plt.show()
plt.clf()

y_pred_probs = best_model.predict_proba(x_test)
y_pred_probs = y_pred_probs[:,1]
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs)
label = "Logistic Regression" +": AP=" +  str(round(metrics.auc(recall, precision),2))
plt.plot(recall, precision, label = label)
plt.legend(loc = "lower center")
plt.title("AUPRC")
plt.show()
plt.clf()  


model = LogisticRegression(solver = "liblinear")
model.fit(x_test, y_test)
y_pred_probs = model.predict_proba(x_test)
y_pred_probs = y_pred_probs[:,1]
fop, mpv = calibration_curve(y_test, y_pred_probs, n_bins=10)
# plot perfectly calibrated
pyplot.plot([0, 1], [0, 1], linestyle='--', color = 'black')
# plot model reliability
pyplot.plot(mpv, fop, marker='.', color = 'red')
pyplot.title("Calibration curve for LR")
pyplot.show()

                
                
                
                
                
                
                
