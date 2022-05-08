# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 10:24:45 2022

@author: rh43233
"""
#pip install keras2onnx==1.7.0
#pip install shap==0.37
#pip install keras2onnx==1.7.0

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, Conv1D, Flatten,Dropout,Activation,MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing import image


import sklearn 
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler
import shap
import graphviz
import pydot
import os
#Changinf the working directory
os.getcwd()
path = "C:/Users/rh43233/OneDrive - The James Hutton Institute/Desktop/JNJ"
os.chdir(path)


#Exploratory Data Analysis
#Data preparation
#Load data

path1= "C:/Users/rh43233/OneDrive - The James Hutton Institute/Desktop/JNJ/Moisture.xlsx" 
FTIR_data_cal = pd.read_excel(path1, sheet_name = 'Cal-SG1')

FTIR_data_pred = pd.read_excel(path1 , sheet_name = 'Pred-SG1')

#checking shape
FTIR_data_cal.shape
FTIR_data_pred.shape
FTIR_data_cal.head()


#set LC as index
FTIR_data_cal.set_index('LC', inplace=True)
FTIR_data_pred.set_index('LC', inplace=True)

#Find number of NA cal samples
FTIR_data_cal.isnull().values.any()

#Find number of NA pred samples
FTIR_data_pred.isnull().values.any()


#print the rows with NA value
null_data = FTIR_data_cal[FTIR_data_cal.isnull().any(axis=1)]
null_data

#Sorting data
Y_cal = FTIR_data_cal[FTIR_data_cal.columns[0]]
X_cal = FTIR_data_cal[FTIR_data_cal.columns[1:1867]]
Y_pred = FTIR_data_pred[FTIR_data_pred.columns[0]]
X_pred = FTIR_data_pred[FTIR_data_pred.columns[1:1867]]


#Standardizing the Data

scaler = StandardScaler()
X_cal2 = scaler.fit_transform(X_cal)
X_pred2 = scaler.fit_transform(X_pred)

#reshape the input data to feed into CNN1D model
X_cal2 = np.expand_dims(X_cal2, axis=2)
X_pred2 = np.expand_dims(X_pred2, axis=2)



model = Sequential()

#Create model

droprate = 0
poolsize = 2

#1Dlayer
model.add(Conv1D(32, kernel_size = 20, activation = 'relu', input_shape =(X_cal2.shape[1],1)))
#model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size = poolsize))
model.add(Dropout(0))

#2Dlayer
model.add(Conv1D(64, kernel_size = 20, activation = 'relu', input_shape=(X_cal2.shape[1],1)))
#model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size = poolsize))
model.add(Dropout(droprate))


#3Dlayer
model.add(Conv1D(128, kernel_size = 20, activation = 'relu', input_shape=(X_cal2.shape[1],1)))
#model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size = poolsize))
model.add(Dropout(droprate))

model.add(Flatten()) 
model.add(Dense(500,use_bias=False))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('linear'))
model.summary()



model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mean_squared_error'])
earlystopper = EarlyStopping(patience=20, verbose=1)
filepath="weights-improvement-{epoch:02d}-{mean_squared_error:.2f}.hdf5"
checkpointer = ModelCheckpoint(filepath, verbose=1, save_best_only=True)


history = model.fit(X_cal2, Y_cal,
          batch_size=32,epochs=3, verbose=1, validation_split=0.2, callbacks=[earlystopper,checkpointer])




# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(1,1, figsize=(18, 10))
ax.plot(history.history['loss'], color='b', label="Training loss")
ax.plot(history.history['val_loss'], color='r', label="validation loss",axes =ax)
legend = ax.legend(loc='best', shadow=True)

score = model.evaluate(X_pred2, Y_pred, verbose=0)
pred_val = model.predict(X_pred2)
z = np.polyfit(Y_pred, pred_val, 1)
sqrt(mean_squared_error (Y_pred, pred_val))
fig, ax = plt.subplots(figsize=(9, 5))
#Plot the best fit line
ax.plot(np.polyval(z,Y_pred), Y_pred, c='blue', linewidth=1)
#Plot the ideal 1:1 line
ax.plot(Y_pred, Y_pred, color='green', linewidth=1)
plt.xlabel('Predicted Moisture')
plt.ylabel('Measured Moisture')
plt.plot(Y_pred, pred_val, '.')

#Find the most important variables ( variables that had the most contribution in order) using SHAP library
cnn_explain = shap.DeepExplainer(model,X_pred2)
shap_values = cnn_explain.shap_values((X_pred2))

#plot shap important variables
imp_f1 = np.reshape(shap_values[0], (154,1866))
Head_WL = X_pred.columns
g = shap.summary_plot(imp_f1,max_display=10 ,feature_names = Head_WL)
g = shap.summary_plot(imp_f1, X_pred, feature_names = Head_WL, max_display=50)
g = shap.summary_plot(imp_f1,X_pred2, plot_type = 'bar',max_display=50, feature_names = Head_WL)










#CNN2d



import natsort
from natsort import natsorted
from os import listdir
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import cv2
import os

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, Conv2D, Flatten,Dropout,Activation,MaxPooling2D


save_dir =  'C:/Users/rh43233/OneDrive - The James Hutton Institute/Desktop/JNJ/Images'
im=[]
os.chdir(save_dir)
for i in range(len(Y_cal)):
    ax = plt.specgram(X_cal.iloc[i,:],Fs=25)
    plt.grid(None)
    plt.axis('off')
    #ax.set_axis_bgcolor("lightslategray")
    plt.savefig("G" + str(i) + ".jpg" , format="JPG", )



images = []
for filename in natsorted(listdir(save_dir)):
    img_data = image.imread(save_dir + '/' + str(filename))
    img_data = cv2.resize(img_data,(32,32))
    images.append(img_data)
    print('> loaded %s %s' % (filename, img_data.shape))



# scale the raw pixel intensities to the range [0, 1]
images = np.array(images, dtype="float") / 255.0
os.chdir("C:/Users/rh43233/OneDrive - The James Hutton Institute/Desktop/JNJ/Images2")
np.save('img_np',images)
images  = np.load("C:/Users/rh43233/OneDrive - The James Hutton Institute/Desktop/JNJ/Images2")

model = Sequential()
        
  
model.add(Conv2D(32, (3, 3), input_shape=(32,32,3), activation= 'relu', padding='same'))
model.add(MaxPooling2D(pool_size= (2,2) ))
model.add(Dropout(0.2))
        
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2,2)))
model.add(Dropout(0.2))
        
model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2,2)))
model.add(Dropout(0.2))
        
        #model.add(Conv2D(256, (3, 3)))
        #model.add(BatchNormalization())
        #model.add(activation=params['activation'])
        #model.add(MaxPooling2D(pool_size= (2,2)))
       # model.add(Dropout = params['DO'])
        
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
#model.add(Dropout(params['DO']))
model.add(Dense(1))
model.add(Activation('linear'))
        
     
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mean_squared_error'])
earlystopper = EarlyStopping(patience=20, verbose=1)

print(model.summary())


file_path ="weights.{epoch:02d}-{val_loss:.2f}.hdf5" 
best_model = ModelCheckpoint(file_path, 
                                     monitor='val_mse', mode='min',verbose=1, 
                                     save_best_only=True)
        
        #earlystopper = EarlyStopping(patience=20, verbose=1)
       # input_shape=(None,x_train.shape[1],x_train.shape[1],3)
       # model.build(input_shape)
history = model.fit(images, Y_cal, validation_split=0.2, epochs=10, batch_size=2)

# Plot the loss and accuracy curves for training and validation 
fig2, ax2 = plt.subplots(1,1, figsize=(18, 10))
ax2.plot(history.history['loss'], color='b', label="Training loss")
ax2.plot(history.history['val_loss'], color='r', label="validation loss",axes =ax2)
legend = ax2.legend(loc='best', shadow=True)



#convert pred dataset to image

save_dir2 =  'C:/Users/rh43233/OneDrive - The James Hutton Institute/Desktop/JNJ/Images_pred'
im2=[]
os.chdir(save_dir2)
for i in range(len(Y_pred)):
    ax = plt.specgram(X_pred.iloc[i,:],Fs=25)
    plt.grid(None)
    plt.axis('off')
    #ax.set_axis_bgcolor("lightslategray")
    plt.savefig("G" + str(i) + ".jpg" , format="JPG", )


image2 = []
for filename in natsorted(listdir(save_dir2)):
    img_data = image.imread(save_dir2 + '/' + str(filename))
    img_data = cv2.resize(img_data,(32,32))
    image2.append(img_data)
    print('> loaded %s %s' % (filename, img_data.shape))

# scale the raw pixel intensities to the range [0, 1]
images_pred = np.array(image2, dtype="float") / 255.0
os.chdir("C:/Users/rh43233/OneDrive - The James Hutton Institute/Desktop/JNJ/Images2")
np.save('img_np_pred',images_pred)




score = model.evaluate(images_pred, Y_pred, verbose=0)
pred_val = model.predict(images_pred)
z = np.polyfit(Y_pred, pred_val, 1)
sqrt(mean_squared_error (Y_pred, pred_val))
fig, ax = plt.subplots(figsize=(9, 5))
#Plot the best fit line
ax.plot(np.polyval(z,Y_pred), Y_pred, c='blue', linewidth=1)
#Plot the ideal 1:1 line
ax.plot(Y_pred, Y_pred, color='green', linewidth=1)
plt.xlabel('Predicted Moisture')
plt.ylabel('Measured Moisture')
plt.plot(Y_pred, pred_val, '.')
