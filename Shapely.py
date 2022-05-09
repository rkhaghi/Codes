# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:43:20 2019

@author: rh43233
"""

def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
from numpy import loadtxt
import keras
from keras.models import Sequential
from keras.layers import Dense
from IPython.display import display, HTML
import shap
import h5py
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt



path = "M:/NSIS2/NSIS2-All data - NIR -FTIR/CNN/final_models"
os.getcwd()
os.chdir(path)



model = load_model('model_carbon_df2.h5')

model = load_model('model_clay.h5')
FTIR_data = pd.read_excel('M:/NSIS2/NSIS2-All data - NIR -FTIR/PLS ,SVR & Cubist results/Clay/NIR/Cubist/Data-Clay-NIR.xlsx'
                          
                          , sheet_name = 'Pred-SG1')



Y = FTIR_data[FTIR_data.columns[1]]
X = FTIR_data[FTIR_data.columns[2:]]
X2 = np.expand_dims(X, axis=2)



cnn_explain = shap.DeepExplainer(model,X2)
shap_values = cnn_explain.shap_values((X2))
imp_f1 = np.reshape(shap_values[0], (154,1866))

Head_WL = X.columns
pp=model.predict(X2)
plt.plot(Y,pp,'.')
g = shap.summary_plot(imp_f1, plot_type = 'bar',max_display=50, feature_names = Head_WL)

g = shap.summary_plot(imp_f1, X, feature_names = Head_WL, max_display=50)

m = imp_f1.shape[0]
n = imp_f1.shape[1]

count_pos = np.zeros(n)

pos_neg2 = np.zeros((2,n))





imp_f1_abs = abs(imp_f1) 
imp_f1_abs_mean = np.zeros((1,n))
for i in range(0,n-1):
    
    imp_f1_abs_mean[0,i] = np.mean(imp_f1_abs[:,i])

X_mean = np.zeros((1,1866))
for i in range(0,1865):
    X_mean[0,i] = np.mean(X.iloc[:,i])
    
    
    
    min_v = np.min(X_mean)
    X_mean_offset = X_mean - min_v 
    max_v = np.max(X_mean_offset)
    X_mean_offset_normalized = np.divide(X_mean_offset,max_v)
    
    fig, ax = plt.subplots()

WN = pd.to_numeric(Head_WL)
ax.bar(WN,imp_f1_abs_mean[0,:], width=20, color = 'red')

plt.clf()

plt.bar(WN,np.multiply(imp_f1_abs_mean[0,:],1), width=20, color = 'red')
plt.xlim(4000,400)
plt.ylim(0,1.05)
plt.xlabel('Wavenumber,$cm^{-1}$', fontsize = 12)
plt.ylabel('Average impact on model outputs', fontsize = 12)
plt.title('Clay', fontsize = 14, fontweight="bold")
plt.xticks(np.arange(4000, 400, step = -200))
plt.yticks(np.arange(0,1.05,step=0.25))
plt.rcParams["figure.figsize"] = (10,6)
plt.plot(WN,np.transpose(  X_mean_offset_normalized), color = 'black')
plt.show()


plt.bar(WN,pos_neg2[0,:])
plt.xlim(4000,400)
plt.show()