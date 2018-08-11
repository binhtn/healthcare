# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 11:37:14 2016

@author: binh
"""

import os
import pandas as pd
from pandas import DataFrame as df
import calendar
import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.cluster import KMeans

from sklearn import preprocessing, tree, svm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer , OneHotEncoder, LabelEncoder
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn import cross_validation, metrics

from sklearn.metrics import recall_score, precision_score, average_precision_score
from sklearn.metrics import confusion_matrix
import numpy.ma as ma


os.chdir('C:/Users/binh/Dropbox (CareSkore)/Customer_data/AIDSHealth/Data/csv')


#==============================================================================
# NOSHOW PREDICTION --  full sample 
#==============================================================================

#to load the newest final_v2 data
ahf = pd.read_pickle('ahf_v3.p')


target = 'cancel'
features = [
            'age_40', 'age_55', 'age_75',  # 'age_75',       
            'female', 
           'other_race',  'white',    'black',            
      #      'divorce',   'single' ,  # 'married',                  
            'perc_noshow' , 'prior_noshow' , 'perc_cancel' , 'prior_cancel',
            'medicare', 'medicaid', 'private',
      #      'distance_30', 'distance_90',  # 'distance_10', 
            'encounter_AM', 'encounter_winter', 
            'duration_15',  'duration_20',  'duration_30',
             'Fri',  # 'Mon', 'Tues', 'Weds', 'Thurs',
     #       'schedule_1w' , 'schedule_4w', 'schedule_6w',  
     #       'prev_appt_3' , 'prev_appt_15', 'prev_appt_30', 
            'middle' , 'rich', 'poor',
            'followup' , 'phlebotomy' , 'othervisit' , 
      #      'college', 'col_above',        #'hsc' ,
           ]
       

##=====
# summary statistics
##=====
# unique patients
len(np.unique(ahf['fldPMPPID']))

##===
# set up training and testing data
##==-
NRow = ahf.shape[0]
Fraction = 0.9    # meaning the training sample is 90% of total sample, testing sample is 10%
np.random.seed(123)        # Setting the random seed
NTrain = int(round(NRow*Fraction))
IndexTrain = np.random.choice(np.arange(NRow), NTrain, replace=False)
Final_Train = ahf.loc[IndexTrain]
Final_Test = ahf.drop(ahf.index[IndexTrain])

XTrain = Final_Train[features]
YTrain = Final_Train[target]

XTest = Final_Test[features]
YTest = Final_Test[target]
 
imp = Imputer(missing_values='NaN', strategy="mean")
            
XTrain_Imp = imp.fit_transform(XTrain)
YTrain_Imp = imp.fit_transform(YTrain.reshape(-1, 1)).ravel().astype(int)   

XTest_Imp = imp.fit_transform(XTest)
YTest_Imp = imp.fit_transform(YTest.reshape(-1, 1)).ravel().astype(int)


######################
### Logistic regression  ##
#####################
noshow_logit = LogisticRegression(penalty='l2',
                                   dual=False, 
                                   tol = 0.0001, 
                                   C=1.0, 
                                   fit_intercept=True, 
                                   intercept_scaling=1, 
                                   class_weight=None, 
                                   random_state=None, 
                                   solver='liblinear', 
                                   max_iter=100, 
                                   multi_class='ovr', 
                                   verbose=0, 
                                   warm_start=False, 
                                   n_jobs=-1)
                                   
noshow_logit.fit(XTrain_Imp, YTrain_Imp)

Acc_Train= noshow_logit.score(XTrain_Imp, YTrain_Imp)
print('Accuracy score, train: %s' %Acc_Train)

Acc_Test= noshow_logit.score(XTest_Imp, YTest_Imp)
print('Accuracy, test: %s' %Acc_Test)


## predict Yhat
LogitPred = noshow_logit.predict_proba(XTest_Imp)

Out = pd.DataFrame()
Out['PatientID'] = Final_Test['fldPMPPID']
Out['YTest'] = YTest
Out['LogitPred'] = LogitPred[:,1]
#   Out.to_csv('LogitPred_noshow_v1.csv')


# export the coefficient
logit_export = pd.DataFrame()
logit_export['feature'] = XTest.columns.values
logit_export['coef'] = noshow_logit.coef_.flatten()
print(logit_export)
#   logit_export.to_csv('logit_export.csv')

print('intercept: %s ' %noshow_logit.intercept_)

#predict yhat
Y_pred1 = noshow_logit.predict_proba(XTest_Imp)[:,1]

# area under the curve
auc_logit= metrics.roc_auc_score(YTest_Imp, Y_pred1)
print('area under the curve: %s' %auc_logit)


#========================================================================
## calculated coefficients and probability manually == exactly the same =
##=======================================================================

LogitPred = noshow_logit.predict_proba(XTest_Imp)[:,1]
coef_list= noshow_logit.coef_.flatten()   
intercept_fit = noshow_logit.intercept_[0]

## alpha=0.0001 is the standard to create the calculated formular exactly as the sklearn model
alpha = 0.00000

scores = intercept_fit + (coef_list[np.newaxis,:]*XTest_Imp).sum(axis=1)
l2_penalty = 2.0*alpha*coef_list.sum()
Out = pd.DataFrame()
Out['fldPMPPID'] = Final_Test['fldPMPPID']
Out['fldAppointmentsID'] = Final_Test['fldAppointmentsID']
Out['YTest'] = YTest
Out['LogitPred'] = LogitPred

Out['cal_score'] = 1./(1. + np.exp(-scores)) - l2_penalty
Out.to_csv('LogitPred_noshow_compare.csv')

## comprare the area under the curve
score_pred= Out['cal_score']
auc_score= metrics.roc_auc_score(YTest_Imp, score_pred)
print('calculated logistic regression roc: ')
print(auc_score)



#=============================================
## How to calculate the optimum threshold 
##============================================

# define high/low threshold
def cal_high_low(var):
    mean = np.mean(var, axis=0)
    std = np.std(var, axis=0)
    high = mean + std/2
    low = mean - std/2
    return high, low
 
## create new dummy variables high/med/low within the dataframe 
def var_high_low (high, low, df, var):
    df['predict_high'] = df[var].map(lambda x : 1 if x > high else 0)
    df['predict_med'] = df[var].map(lambda x : 1 if x <= high and x > low else 0)
    df['predict_low'] = df[var].map(lambda x : 1 if x <= low else 0)
    return  df
    
## only do it with testing sample, not training sample 
    
high, low = cal_high_low(Out['LogitPred'])    
var_high_low (high, low, Out, 'LogitPred')

print ('High threshold: %s' %high) 
print ('Low threshold: %s' %low) 


## play with the threshold  

# high threshold
best_threshold = threshold_values[np.argmax(ma.masked_invalid(AllF1))]
print ('Best threshold: %s' %best_threshold) 


ApplyThreshold_Single  = lambda Prob, Threshold: (Prob > Threshold).astype(int)
Threshold = high

Y_pred1_binary = ApplyThreshold_Single(Y_pred1, Threshold)
   
rec_score = recall_score(YTest_Imp, Y_pred1_binary)  
print('recall score, test: %s' %rec_score)


pre_score = precision_score(YTest_Imp, Y_pred1_binary) 
print('precision score, test: %s' %pre_score)


## calculate the confusion matrix
cmat = confusion_matrix(y_true=YTest_Imp,
                        y_pred= Y_pred1_binary,
                        labels=noshow_logit.classes_)    # use the same order of class as the LR model.
                        
print (' target_label | predicted_label | count ')
print ('--------------+-----------------+-------')
# Print out the confusion matrix.
for i, target_label in enumerate(noshow_logit.classes_):
    for j, predicted_label in enumerate(noshow_logit.classes_):
        print ('{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[i,j]))
        
# caculate true positive
TN = cmat[0][0]
FP = cmat[0][1]
FN = cmat[1][0]
TP = cmat[1][1]

recall = TP/(TP+FN)
print('recall score, calculated: %s' %recall)

precision = TP/(TP+FP)
print('precision score, calculated: %s' %precision)

highrisk_pop = (FP + TP)/(TN + FP + FN  + TP)
print('Percent of population predicted highrisk, calculated: %s' %highrisk_pop)


#===============
# low threshold
#================

best_threshold = threshold_values[np.argmax(ma.masked_invalid(AllF1))]
print ('Best threshold: %s' %best_threshold) 


ApplyThreshold_Single  = lambda Prob, Threshold: (Prob > Threshold).astype(int)
Threshold = low

Y_pred1_binary = ApplyThreshold_Single(Y_pred1, Threshold)
   
rec_score = recall_score(YTest_Imp, Y_pred1_binary)  
print('recall score, test: %s' %rec_score)


pre_score = precision_score(YTest_Imp, Y_pred1_binary) 
print('precision score, test: %s' %pre_score)


## calculate the confusion matrix
cmat = confusion_matrix(y_true=YTest_Imp,
                        y_pred= Y_pred1_binary,
                        labels=noshow_logit.classes_)    # use the same order of class as the LR model.
                        
print (' target_label | predicted_label | count ')
print ('--------------+-----------------+-------')
# Print out the confusion matrix.
for i, target_label in enumerate(noshow_logit.classes_):
    for j, predicted_label in enumerate(noshow_logit.classes_):
        print ('{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[i,j]))
        
# caculate true positive
TN = cmat[0][0]
FP = cmat[0][1]
FN = cmat[1][0]
TP = cmat[1][1]

recall = TP/(TP+FN)
print('recall score, calculated: %s' %recall)

precision = TP/(TP+FP)
print('precision score, calculated: %s' %precision)

highrisk_pop = (FP + TP)/(TN + FP + FN  + TP)
print('Percent of population predicted high & medium risk, calculated: %s' %highrisk_pop)




# optimum threshold for recall  
threshold_values = np.linspace(0.0, 1.0, num=300)
print (threshold_values)
ApplyThreshold = lambda Prob, AllThreshold: (Prob[:, np.newaxis] > AllThreshold[np.newaxis, :]).astype(int)

Ypred_binary_multi = ApplyThreshold(LogitPred[:,1], threshold_values)
AllRecall = np.array([recall_score(YTest_Imp, Ypred_binary_multi[:, i]) for i in np.arange(Ypred_binary_multi.shape[1])])
AllPrecision = np.array([precision_score(YTest_Imp, Ypred_binary_multi[:, i]) for i in np.arange(Ypred_binary_multi.shape[1])])

AllF1 = AllRecall*AllPrecision/(AllRecall+AllPrecision)*2.0
maxf1_threshold =  threshold_values[np.argmax(ma.masked_invalid(AllF1))]      # threshold maximize F1 score

cmat = confusion_matrix(y_true=YTest_Imp,
                        y_pred=Ypred_binary_multi[:,maxf1_threshold],
                        labels=noshow_logit.classes_)    # use the same order of class as the LR model.                    
print (' target_label | predicted_label | count ')
print ('--------------+-----------------+-------')
for i, target_label in enumerate(noshow_logit.classes_):
        for j, predicted_label in enumerate(noshow_logit.classes_):
            print ('{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[i,j]))
  
            
            
def best_threshold (predicted_y, YTest_Imp , model):
    threshold_values = np.linspace(0.0, 1.0, num=200)
    ApplyThreshold = lambda Prob, AllThreshold: (Prob[:, np.newaxis] > AllThreshold[np.newaxis, :]).astype(int)
    Ypred_binary_multi = ApplyThreshold(predicted_y, threshold_values)
    AllRecall = np.array([recall_score(YTest_Imp, Ypred_binary_multi[:, i]) for i in np.arange(Ypred_binary_multi.shape[1])])
    AllPrecision = np.array([precision_score(YTest_Imp, Ypred_binary_multi[:, i]) for i in np.arange(Ypred_binary_multi.shape[1])])
    AllF1 = AllRecall*AllPrecision/(AllRecall+AllPrecision)*2.0
    maxf1_threshold = threshold_values[np.argmax(ma.masked_invalid(AllF1))]     
    cmat = confusion_matrix(y_true=YTest_Imp,
                        y_pred=Ypred_binary_multi[:,maxf1_threshold],
                        labels=model.classes_)    # use the same order of class as the LR model.                    
    print (' target_label | predicted_label | count ')
    print ('--------------+-----------------+-------')
    for i, target_label in enumerate(model.classes_):
        for j, predicted_label in enumerate(model.classes_):
            print ('{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[i,j]))
    return maxf1_threshold , cmat                

    
 

## plot all the figures, plot AUC too // need to work on this
false_positive_rate = FP/(FP + TN)
true_positive_rate = TP/(TP + FN)

# This is the ROC curve
plt.plot(false_positive_rate,true_positive_rate, 'ro')
plt.show() 

# This is the AUC
area_under_curve = np.trapz(true_positive_rate,false_positive_rate)


plt.figure()
plt.plot(AllPrecision, AllRecall, 'ro')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.show()
plt.clf()
plt.cla()
plt.close('all')


def plot_pr_curve(threshold_values, AllF1 , title):
    plt.rcParams['figure.figsize'] = 7, 5
    plt.locator_params(axis = 'x', nbins = 20)
    plt.plot(threshold_values, AllF1, 'b-', linewidth=4.0, color = '#B0017F')
    plt.title(title)
    plt.xlabel('Threshold Values')
    plt.ylabel('F1 score')
    plt.rcParams.update({'font.size': 12})
     
plot_pr_curve(threshold_values, AllF1, 'Precision recall curve (all)') 
     
   
    
##============================
##     RANDOM FOREST        ===
##============================     

forest = RandomForestClassifier(n_estimators=200,   # change here. from 10
                                criterion='gini',
                                max_depth=10,       # change here, from None
                                min_samples_split=2, 
                                min_samples_leaf=1, 
                                min_weight_fraction_leaf=0.0, 
                                max_features='auto', 
                                max_leaf_nodes=None, 
                                bootstrap=True, oob_score=False, 
                                n_jobs=1, 
                                random_state=None, 
                                verbose=0, 
                                warm_start=False, 
                                class_weight=None)
    
forest.fit(XTrain_Imp, YTrain_Imp)

Acc_Train= forest.score(XTrain_Imp, YTrain_Imp)
print('accuracy, train:')
print(Acc_Train)


Acc_Test= forest.score(XTest_Imp, YTest_Imp)
print('accuracy, test:')
print(Acc_Test)

## predict Yhat
ForestPred = forest.predict_proba(XTest_Imp)[:,1]

Out = pd.DataFrame()
Out['PatientID'] = Final_Test['fldPMPPID']
Out['YTest'] = YTest
Out['ForestPred'] = ForestPred
Out.to_csv('ForestPred_noshow_v1.csv')

# Build a forest and compute the feature importances
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

FeatureRanking = pd.DataFrame(features, columns=['Feature'])
FeatureRanking['Importance'] = importances
FeatureRanking = FeatureRanking.loc[indices]
FeatureRanking.to_csv('FR_RF_noshow.csv')

# area under the curve
auc_rf= metrics.roc_auc_score(YTest_Imp, ForestPred )
print('Area under the curve - Random Forest %s' %auc_rf)

# no show: Area under the curve - Random Forest 0.796560002544


#======================
# K NEAREST NEIGHBOR  =
#======================
from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier (n_neighbors=2 ,              #default: 5
                            weights='uniform', 
                            algorithm='auto', 
                            leaf_size=30, 
                            p=2, 
                            metric='minkowski', 
                            metric_params=None, 
                            n_jobs=1)

knc = knc.fit(XTrain_Imp, YTrain_Imp)

knc_pred = knc.predict_proba(XTest_Imp)[:,1]
auc_knc = metrics.roc_auc_score (YTest_Imp, knc_pred)

print ('area under the curve for K nearest neighbor %s' %auc_knc) 

Acc_Train= knc.score(XTrain_Imp, YTrain_Imp)
print('accuracy, train: %s' %Acc_Train )


Acc_Test= knc.score(XTest_Imp, YTest_Imp)
print('accuracy, test: %s' %Acc_Test )


Out = pd.DataFrame()
Out['PatientID'] = Final_Test['fldPMPPID']
Out['YTest'] = YTest
Out['ForestPred'] = knc_pred
Out.to_csv('KNN_noshow.csv')


# optimum threshold for recall  
threshold_values = np.linspace(0.0, 1.0, num=200)
print (threshold_values)
ApplyThreshold = lambda Prob, AllThreshold: (Prob[:, np.newaxis] > AllThreshold[np.newaxis, :]).astype(int)

Ypred_knc_binary_multi = ApplyThreshold(knc_pred, threshold_values)
AllRecall = np.array([recall_score(YTest_Imp, Ypred_knc_binary_multi[:, i]) for i in np.arange(Ypred_knc_binary_multi.shape[1])])
AllPrecision = np.array([precision_score(YTest_Imp, Ypred_knc_binary_multi[:, i]) for i in np.arange(Ypred_knc_binary_multi.shape[1])])

AllF1 = AllRecall*AllPrecision/(AllRecall+AllPrecision)*2.0
threshold_values[np.argmax(ma.masked_invalid(AllF1))]      # threshold maximize F1 score

## calculate the confusion matrix for optimal F1
cmat = confusion_matrix(y_true=YTest_Imp,
                        y_pred=Ypred_knc_binary_multi[:,np.argmax(ma.masked_invalid(AllF1))],
                        labels=knc.classes_)    # use the same order of class as the LR model.
                        
print (' target_label | predicted_label | count ')
print ('--------------+-----------------+-------')
# Print out the confusion matrix.
for i, target_label in enumerate(knc.classes_):
    for j, predicted_label in enumerate(knc.classes_):
        print ('{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[i,j]))
        
        
       
   
## play with the threshold  

# optimum threshold for recall  
threshold_values[np.argmax(ma.masked_invalid(AllF1))]

ApplyThreshold_Single  = lambda Prob, Threshold: (Prob > Threshold).astype(int)
Threshold = 0.0717908

Y_pred1_binary = ApplyThreshold_Single(Y_pred1, Threshold)
   
rec_score = recall_score(YTest_Imp, Y_pred1_binary)  
print('recall score, test:')
print(rec_score)


pre_score = precision_score(YTest_Imp, Y_pred1_binary) 
print('precision score, test:')
print(pre_score)


## calculate the confusion matrix
cmat = confusion_matrix(y_true=YTest_Imp,
                        y_pred= Y_pred1_binary,
                        labels=noshow_logit.classes_)    # use the same order of class as the LR model.
                        
print (' target_label | predicted_label | count ')
print ('--------------+-----------------+-------')
# Print out the confusion matrix.
for i, target_label in enumerate(noshow_logit.classes_):
    for j, predicted_label in enumerate(noshow_logit.classes_):
        print ('{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[i,j]))
        

## plot all the figures 
plt.figure()
plt.plot(AllPrecision, AllRecall, 'ro')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.show()
plt.clf()
plt.cla()
plt.close('all')


def plot_pr_curve(threshold_values, AllF1 , title):
    plt.rcParams['figure.figsize'] = 7, 5
    plt.locator_params(axis = 'x', nbins = 20)
    plt.plot(threshold_values, AllF1, 'b-', linewidth=4.0, color = '#B0017F')
    plt.title(title)
    plt.xlabel('Threshold Values')
    plt.ylabel('F1 score')
    plt.rcParams.update({'font.size': 12})
     
plot_pr_curve(threshold_values, AllF1, 'Precision recall curve (all)') 
     

##=====================================
# Use STERN coefficients for AHF data
#======================================
# to load 
import pickle

with open('C:/Users/binh/Dropbox (CareSkore)/Customer_data/STERN/STERN_csv/stern_model.pkl', 'rb') as f:
    stern_model = pickle.load(f)

                            
# stern_model.fit(XTrain_Imp, YTrain_Imp)
Acc_Train= stern_model.score(XTrain_Imp, YTrain_Imp)
print('Accuracy score, train: %s' %Acc_Train)

Acc_Test= stern_model.score(XTest_Imp, YTest_Imp)
print('Accuracy, test: %s' %Acc_Test)


## predict Yhat
LogitPred = stern_model.predict_proba(XTest_Imp)
LogitPred_t = stern_model.predict_proba(XTrain_Imp)

Out = pd.DataFrame()
Out['PatientID'] = Final_Test['fldPMPPID']
Out['YTest'] = YTest
Out['LogitPred'] = LogitPred[:,1]
#   Out.to_csv('LogitPred_noshow_v1.csv')


# export the coefficient
logit_export = pd.DataFrame()
logit_export['feature'] = XTest.columns.values
logit_export['coef'] = stern_model.coef_.flatten()
print(logit_export)
#   logit_export.to_csv('logit_export.csv')

print('intercept: %s ' %stern_model.intercept_)

#predict yhat
Y_pred1= stern_model.predict_proba(XTest_Imp)[:,1]

# area under the curve
auc_logit= metrics.roc_auc_score(YTest_Imp, Y_pred1)
print('area under the curve using STERN coefficients: %s' %auc_logit)




#======================
# K NEAREST NEIGHBOR with grid search =
#======================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

    
k = np.arange(5) + 1
parameters = {'n_neighbors': k}

clf = KNeighborsClassifier().fit(XTrain_Imp, YTrain_Imp)
clf = GridSearchCV(clf, parameters, scoring ='roc_auc')
clf.fit(XTest_Imp, YTest_Imp)
clf.best_estimator_
    
    
# other template for Gridsearch -- problem: it didn't pick the best parameters
# for more:     parameters = {"C":[0.1, 1, 10, 100, 1000], "gamma":[0.1, 0.01, 0.001, 0.0001, 0.00001]}

scores = ['precision', 'recall', 'roc_auc']

for score in scores:
    print("# Tuning hyper-parameters for %s" %score)
    print()
    clf = KNeighborsClassifier().fit(XTrain_Imp, YTrain_Imp)
    clf = GridSearchCV(KNeighborsClassifier(), parameters,cv=10, scoring='%s' %score)
    
    clf.fit(XTest_Imp, YTest_Imp)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = YTest_Imp, clf.predict(XTest_Imp)
    print(classification_report(y_true, y_pred))
    print()


# other example SVC
from sklearn.svm import SVC
estimator = SVC(kernel='linear')
gammas = np.logspace(-6, -1, 10)

from sklearn.cross_validation import ShuffleSplit
cv = ShuffleSplit(XTrain_Imp.shape[0], n_iter=10, test_size=0.2, random_state=0)

classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=dict(gamma=gammas))
classifier.fit(XTrain_Imp, YTrain_Imp)

from sklearn.learning_curve import learning_curve
title = 'Learning Curves (SVM, linear kernel, $\gamma=%.6f$)' %classifier.best_estimator_.gamma
estimator = SVC(kernel='linear', gamma=classifier.best_estimator_.gamma)
learning_curve(estimator, title, XTrain_Imp, YTrain_Imp, cv=cv)
plt.show()

#===========================
# Gradient Boosting Classifier =
#===========================

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence
gradientboost = GradientBoostingClassifier (loss='deviance', 
                                            learning_rate=0.1, 
                                            n_estimators=200,           #change here to be consistent with RF
                                            subsample=1.0, 
                                            min_samples_split=2, 
                                            min_samples_leaf=1, 
                                            min_weight_fraction_leaf=0.0, 
                                            max_depth=5,    # chage from 3 to 6
                                            init=None, 
                                            random_state=None, 
                                            max_features=None, 
                                            verbose=0, 
                                            max_leaf_nodes=None, 
                                            warm_start=False, 
                                            presort='auto').fit(XTrain_Imp, YTrain_Imp)

gradientboost_pred = gradientboost.predict_proba(XTest_Imp)[:,1]
# area under the curve
auc_gradientboost  =  metrics.roc_auc_score(YTest_Imp, gradientboost_pred)
print('gradient boost classification AUC')
print(auc_gradientboost)
## cancellation model 0.727215209056


#===========================
#    AdaBoostClassifier    =
#===========================

from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier (base_estimator=None,
                          n_estimators=50, 
                          learning_rate=1.0, 
                          algorithm='SAMME.R', 
                          random_state=None).fit(XTrain_Imp, YTrain_Imp)

ada_pred = ada.predict_proba(XTest_Imp)[:,1]
# area under the curve
auc_ada  =  metrics.roc_auc_score(YTest_Imp, ada_pred)
print('Ada boost classification AUC')
print(auc_ada)


#===========================
# support vector machine =
#===========================
from sklearn import svm

C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', 
              probability=True,
              C=C).fit(XTest_Imp, YTest_Imp)
svc_pred = svc.predict_proba(XTest_Imp)[:,1]
auc_svc = metrics.roc_auc_score ( YTest_Imp, svc_pred )
print ('area under the curve for SVM linear kernel %s' %auc_svc)          
# horrible 0.493994532478 

# the (Gaussian) radial basis function kernel, or RBF kernel, is a popular kernel function used in various kernelized learning algorithms.
rbf_svc = svm.SVC(kernel='rbf',
                  probability=True,
                  C=C).fit(XTest_Imp, YTest_Imp)
rbf_svc_pred = rbf_svc.predict_proba(XTest_Imp)[:,1]
auc_rbf_svc = metrics.roc_auc_score ( YTest_Imp, rbf_svc_pred )
print ('area under the curve for SVM gaussian radial basis function kernel %s' %auc_rbf_svc)
# high 0.786126650216                 
        
          
poly_svc = svm.SVC(kernel='poly',
                   probability=True,
                   C=C).fit(XTest_Imp, YTest_Imp)
poly_pred = poly_svc.predict_proba(XTest_Imp)[:,1]
auc_poly = metrics.roc_auc_score ( YTest_Imp, poly_pred )
print ('area under the curve for SVM polynomial kernel %s' %auc_poly)
# 0.706882663358, low        


### a default svm       
noshow_svm = svm.SVC(C=1.0, kernel='rbf', 
                     degree=3, 
                     gamma='auto', 
                     coef0=0.0, 
                     shrinking=True, 
                     probability=True,
                     tol=0.001, 
                     cache_size=200,
                     class_weight=None, 
                     verbose=False, 
                     max_iter=-1, 
                     decision_function_shape=None,
                     random_state=None).fit(XTest_Imp, YTest_Imp)
                     
noshow_svm_pred = noshow_svm.predict_proba(XTest_Imp)[:,1]
auc_cancel_svm = metrics.roc_auc_score ( YTest_Imp, noshow_svm_pred )
print ('area under the curve for SVM default option %s' %auc_cancel_svm)
## 0.724839947631   == similar to noshow prediction, svm with rbf kernel is the best. 


## can't calculate the coefficients since this is not linear model. 



##=========================
# NAIVE BAYES GAUSSIAN NB #
##=========================
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB


nb_gaussian = GaussianNB().fit(XTrain_Imp, YTrain_Imp)
nb_bernou = BernoulliNB().fit(XTrain_Imp, YTrain_Imp)


nbg_pred = nb_gaussian.predict_proba(XTest_Imp)[:,1]
nbb_pred = nb_bernou.predict_proba(XTest_Imp)[:,1]


# area under the curve
auc_nbg  =  metrics.roc_auc_score(YTest_Imp, nbg_pred)
auc_nbb  =  metrics.roc_auc_score(YTest_Imp, nbb_pred)


print('Naive Bayes Gaussian classification AUC %s:' %auc_nbg)
print('Naive Bayes BernoulliNB classification AUC %s:' %auc_nbb)



#=====================================
# Voting Classifier / soft voting   ==
#=====================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
eclf = VotingClassifier (estimators=[('dt', clf1), 
                                     ('knn', clf2), 
                                     ('svc', clf3)], 
                                        voting='soft', 
                                        weights=[2,1,2])

clf1.fit(XTrain_Imp, YTrain_Imp)
clf2.fit(XTrain_Imp, YTrain_Imp)
clf3.fit(XTrain_Imp, YTrain_Imp)
eclf.fit(XTrain_Imp, YTrain_Imp)


for clf, label in zip([clf1, clf2, clf3, eclf], ['Decision Tree', 'K Neighbor', 'SVC',  'Ensemble']):
        scores2 = cross_validation.cross_val_score(clf ,
                                                  XTest_Imp , 
                                                  YTest_Imp , 
                                                  cv=5, 
                                                  scoring = 'roc_auc')
                                                  
print("Area under the curve: %0.5f (+/- %0.5f) [%s]" % (scores2.mean(), scores.std(), label))
        



#==============================================
# Using logistic regression with GridSearch ==
#==============================================
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

param_grid = {'C': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000] }
clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
GridSearchCV(cv=None,
             estimator=LogisticRegression (C=1.0, intercept_scaling=1, dual=False, fit_intercept=True,
                                           penalty='l2', tol=0.0001),
                                           param_grid={'C': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]})

clf.fit(XTrain_Imp, YTrain_Imp)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
print()
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
YTrain_Imp, y_pred = YTest_Imp, clf.predict(XTest_Imp)
print(classification_report(YTest_Imp, y_pred))
print()