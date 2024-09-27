from pandas import read_csv
#import tensorflow.keras as keras
import keras
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import sklearn
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import numpy as np     
import csv 
import copy 
import random 
import pandas
import matplotlib.pyplot as plt 
from sklearn.preprocessing import normalize



##  Read Data  ##

ifile  = open('Data.csv', "rt")
reader = csv.reader(ifile)
csvdata=[]
for row in reader:
        csvdata.append(row)   
ifile.close()
numrow=len(csvdata)
numcol=len(csvdata[0]) 
csvdata = np.array(csvdata).reshape(numrow,numcol)
Sys = csvdata[:,0]
Atom_A = csvdata[:,1]
Atom_B = csvdata[:,2]
Struct = csvdata[:,3]
Vol_pfu = csvdata[:,4]
Delta_H = csvdata[:,5]
Gap = csvdata[:,6]
X = csvdata[:,7:]

    # Read Outside Data
#ifile  = open('Outside.csv', "rt")
#reader = csv.reader(ifile)
#csvdata=[]
#for row in reader:
#        csvdata.append(row)
#ifile.close()
#numrow=len(csvdata)
#numcol=len(csvdata[0])
#csvdata = np.array(csvdata).reshape(numrow,numcol)
#Sys_out = csvdata[:,0]
#Atom_A_out = csvdata[:,1]
#Atom_B_out = csvdata[:,2]
#X_out = csvdata[:,3:]

#n_out = Sys_out.size


X_fl = np.array(X, dtype="float32")
X_norm = normalize(X_fl, norm='l2', axis=0)

Prop = copy.deepcopy(Delta_H)
n = Sys.size
m = int(X.size/n)

t = 0.20

X_train, X_test, Prop_train, Prop_test, Atom_A_train, Atom_A_test, Atom_B_train, Atom_B_test  =  train_test_split(X_norm, Prop, Atom_A, Atom_B, test_size=t)

n_tr = Prop_train.size
n_te = Prop_test.size

X_train_fl = np.array(X_train, dtype="float32")
X_test_fl = np.array(X_test, dtype="float32")
Prop_train_fl = np.array(Prop_train, dtype="float32")
Prop_test_fl = np.array(Prop_test, dtype="float32")





###  NN Optimizers and Model Definition  ###

pipelines = []

parameters = [[0.0 for a in range(6)] for b in range(729)]

dp = [0.00, 0.10, 0.20]
n1 = [50, 75, 100]
n2 = [50, 75, 100]
lr = [0.001, 0.01, 0.1]
ep = [200, 400, 600]
bs = [50, 100, 200]

count = 0
for a in range(0,3):
    for b in range(0,3):
        for c in range(0,3):
            for d in range(0,3):
                for e in range(0,3):
                    for f in range(0,3):
                        parameters[count][0] = lr[a]
                        parameters[count][1] = n1[b]
                        parameters[count][2] = dp[c]
                        parameters[count][3] = n2[d]
                        parameters[count][4] = ep[e]
                        parameters[count][5] = bs[f]
                        count = count+1
                        
                        keras.optimizers.Adam(learning_rate=lr[a], beta_1=0.9, beta_2=0.999, amsgrad=False)
                        
                        # define base model
                        def baseline_model():
                            model = Sequential()
                            model.add(Dense(m, input_dim=m, kernel_initializer='normal', activation='relu'))
                            model.add(Dense(n1[b], kernel_initializer='normal', activation='relu'))
                            model.add(Dropout(dp[c], input_shape=(m,)))
                            model.add(Dense(n2[d], kernel_initializer='normal', activation='relu'))
                            model.add(Dense(1, kernel_initializer='normal'))
                            model.compile(loss='mean_squared_error', optimizer='Adam')
                            return model
                        # evaluate model with standardized dataset
                        estimators = []
#                        estimators.append(('standardize', StandardScaler()))
                        estimators.append(('scaler', StandardScaler()))
                        estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=ep[e], batch_size=bs[f], verbose=0)))
                        pipelines.append ( Pipeline(estimators) )

times = 1
#times = len(pipelines)




  ##  Train Neural Network Model  ##


train_errors = [0.0]*times
test_errors = [0.0]*times
nn_errors = list()

n_fold = 5

for i in range(0,times):
    pipeline = pipelines[np.random.randint(0,729)]
#    pipeline = pipelines[i]
    kf = KFold(n_splits = n_fold)
    mse_test_cv = 0.00
    mse_train_cv = 0.00
    for train, test in kf.split(X_train):
        X_train_cv, X_test_cv, Prop_train_cv, Prop_test_cv = X_train[train], X_train[test], Prop_train[train], Prop_train[test]

        X_train_cv_fl = np.array(X_train_cv, dtype="float32")
        Prop_train_cv_fl = np.array(Prop_train_cv, dtype="float32")
        X_test_cv_fl = np.array(X_test_cv, dtype="float32")
        Prop_test_cv_fl = np.array(Prop_test_cv, dtype="float32")

        pipeline.fit(X_train_cv_fl, Prop_train_cv_fl)
        Prop_pred_train_cv = pipeline.predict(X_train_cv_fl)
        Prop_pred_test_cv  = pipeline.predict(X_test_cv_fl)
        Pred_train_cv_fl = np.array(Prop_pred_train_cv, dtype="float32")
        Pred_test_cv_fl = np.array(Prop_pred_test_cv, dtype="float32")

        mse_test_cv = mse_test_cv  + sklearn.metrics.mean_squared_error(Prop_test_cv_fl, Pred_test_cv_fl)
        mse_train_cv = mse_train_cv + sklearn.metrics.mean_squared_error(Prop_train_cv_fl, Pred_train_cv_fl)
    mse_test = mse_test_cv / n_fold
    mse_train = mse_train_cv / n_fold
    train_errors[i] = mse_train
    test_errors[i] = mse_test
    nn_errors.append(pipeline)
i_opt = np.argmin(test_errors)
pipeline_opt = nn_errors[i_opt]

pipeline_opt.fit(X_train_fl, Prop_train_fl)
Pred_train = pipeline_opt.predict(X_train_fl)
Pred_test = pipeline_opt.predict(X_test_fl)

Pred_train_fl = np.array(Pred_train, dtype="float32")
Pred_test_fl = np.array(Pred_test, dtype="float32")


## Outside Predictions

#Pred_out = pipeline_opt.predict(X_out_fl)
#for i in range(0,n_out):
#    Pred_out_fl[i][1] = float(Pred_out[i])


mse_test = sklearn.metrics.mean_squared_error(Prop_test_fl, Pred_test_fl)
mse_train = sklearn.metrics.mean_squared_error(Prop_train_fl, Pred_train_fl)
rmse_test = np.sqrt(mse_test)
rmse_train = np.sqrt(mse_train)
print('rmse_test_DH = ', np.sqrt(mse_test))
print('rmse_train_DH = ', np.sqrt(mse_train))
print('      ')





## ML Parity Plots ##

fig = plt.figure(figsize=(6,6))
plt.subplots_adjust(left=0.19, bottom=0.17, right=0.95, top=0.88, wspace=0.25, hspace=0.4)
plt.rc('font', family='Arial narrow')
plt.rcParams.update({'font.size': 16})

plt.ylabel('ML Prediction', fontname='Arial Narrow', size=32)
plt.xlabel('DFT Calculation', fontname='Arial Narrow', size=32, labelpad=10)

a = [-175,0,125]
b = [-175,0,125]
plt.plot(b, a, c='k', ls='-')

plt.scatter(Prop_train_fl[:], Pred_train_fl[:], c='blue', marker='s', s=100, edgecolors='dimgrey', alpha=0.5, label='Training')
plt.scatter(Prop_test_fl[:], Pred_test_fl[:], c='orange', marker='s', s=100, edgecolors='dimgrey', alpha=0.2, label='Test')

#plt.errorbar(Prop_train_fl[:], Pred_train_fl[:], yerr = [err_up_train[:]-Pred_train_fl[:], Pred_train_fl[:]-err_down_train[:]], c='blue', marker='s', markeredgecolor='dimgrey', markersize=10, fmt='o', ecolor='blue', capthick=2, label='Training')
#plt.errorbar(Prop_test_fl[:], Pred_test_fl[:], yerr =  [err_up_test[:]-Pred_test_fl[:], Pred_test_fl[:]-err_down_test[:]], c='orange', marker='s', markeredgecolor='dimgrey', markersize=10, fmt='o', ecolor='orange', capthick=2, label='Test')

te = '%.2f' % rmse_test
tr = '%.2f' % rmse_train

plt.text(1.15, -8.0, 'Train_rmse = ', color='r', fontsize=16)
plt.text(7.0, -8.0, tr, color='r', fontsize=16)
plt.text(9.0, -8.0, 'eV', color='r', fontsize=16)
plt.text(1.5, -9.8, 'Test_rmse = ', color='r', fontsize=16)
plt.text(7.0, -9.8, te, color='r', fontsize=16)
plt.text(9.0, -9.8, 'eV', color='r', fontsize=16)

plt.ylim([-12, 12])
plt.xlim([-12, 12])
plt.xticks([-10, -5, 0, 5, 10])
plt.yticks([-10, -5, 0, 5, 10])

plt.title('Formation Enthalpy (eV)', color='k', fontsize=28, pad=15)
plt.legend(loc='upper left',ncol=1, frameon=True, prop={'family':'Arial narrow','size':20})
plt.rc('xtick', c='k', labelsize=24)
plt.rc('ytick', c='k', labelsize=24)

plt.savefig('plot.eps', dpi=450)
#plt.show()



