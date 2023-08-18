#!/usr/bin/env python
# coding: utf-8

# Required Libraries

from autogluon.tabular import TabularPredictor as task
from collections import Counter
from __future__ import absolute_import, division, print_function, unicode_literals
from matplotlib import pyplot as plt
from pandas import read_csv
from PIL import Image
from random import randint
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage.exposure import equalize_hist
from skimage import color, exposure, transform
from skimage.io import imread, imshow
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import argparse
import autogluon as ag
import autogluon.core as ag
import cv2 as cv
import glob
import logging
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle# for loading/processing the
import PIL.Image as Image
import re
import sklearn
import sys
import tensorflow as tf

# Constants
SEED = "1010"
#path to save autogluon model
save_path="./sample_data/vgg_ag"
# Loading Images
circos_images_full = glob.glob("./sample_data/sample/*/*/im.*.png")
circos_images = {}
n=0
for i in circos_images_full:
    n=n+1
    circos_images['l'+str(n)+os.path.basename(i)] = i


keys = list(circos_images.keys())
keys[1:5],circos_images[keys[1]]


cate={}
lbl={}
for i in keys:
    lbl[i]=os.path.basename(os.path.dirname(circos_images[i]))
    cate[i]=os.path.basename(os.path.dirname(os.path.dirname(circos_images[i])))






model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
# Extracting features for each image
def extract_features(file, model):
    img = cv.imread(file, cv.IMREAD_UNCHANGED)#cv2.IMREAD_COLOR)IMREAD_UNCHANGED
    img = cv.resize(img, (224,224), interpolation = cv.INTER_CUBIC)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features


data = {}
# lop through each image in the dataset
for flower in keys:
    feat = extract_features(circos_images[flower],model)
    data[flower] = feat

filenames = np.array([i for i in keys])
feat = np.array([data[i] for i in keys])
# reshape so that there are 210 samples of 4096 vectors
feat = feat.reshape(-1,4096)
feat.shape,filenames[:5],keys[1:5],len(filenames)
list_lbl = [lbl[i] for i in keys]
list_cate = [cate[i] for i in keys]
list_lbl_cate = [lbl[i]+'_'+cate[i] for i in keys]


Counter(list_lbl_cate)
cate_train = [list_lbl[i] for i in range(0,len(list_cate),1) if list_cate[i] == "train" ]
cate_val = [list_lbl[i] for i in range(0,len(list_cate),1) if list_cate[i] == "val" ]
cate_test = [list_lbl[i] for i in range(0,len(list_cate),1) if list_cate[i] == "test" ]
train_ind = [i for i in range(0,len(list_cate),1) if list_cate[i] == "train" ]
val_ind = [i for i in range(0,len(list_cate),1) if list_cate[i] == "val" ]
test_ind = [i for i in range(0,len(list_cate),1) if list_cate[i] == "test" ]
val_np = feat[val_ind,]
filenm_val = [ filenames[i] for i in val_ind]
test_np = feat[test_ind,]
filenm_test = [ filenames[i] for i in test_ind]
train_np = feat[train_ind,]
filenm_train = [ filenames[i] for i in train_ind]


pca = PCA(n_components=50, random_state=22)
pca.fit(train_np)
x = pca.transform(train_np)
x_val = pca.transform(val_np)
x_test = pca.transform(test_np)


y_ori_train = [ i for i in cate_train ]
y_ori_val = [ i for i in cate_val ]
y_ori_test = [ i for i in cate_test ]

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x, y_ori_train)
y_pred = classifier.predict(x_val)

print(confusion_matrix(y_ori_val, y_pred))

print(classification_report(y_ori_val, y_pred))


y_pred = classifier.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_ori_test, y_pred))
print(classification_report(y_ori_test, y_pred))

y_pred_train = classifier.predict(x)
print(classification_report(y_ori_train, y_pred_train))

colnames=[ "feature"+str(i) for i in range(0,50,1)]

train_df = pd.DataFrame(x, columns = colnames)
test_df = pd.DataFrame(x_test, columns = colnames)
val_df = pd.DataFrame(x_val, columns = colnames)

train_df['label']=y_ori_train
test_df['label']=y_ori_test
val_df['label']=y_ori_val

predictor = task(label='label', path=save_path,eval_metric = "balanced_accuracy").fit(train_data=train_df,tuning_data=val_df)

performance = predictor.evaluate(test_df)
y_test = test_df['label']
test_data_nolab = test_df.drop(columns=['label'])
y_pred = predictor.predict(test_data_nolab)
def report(fish_list, result_list,prob, name):
    tn, fp, fn, tp = metrics.confusion_matrix(fish_list, result_list).ravel()
    tn=int(tn)
    fp=int(fp)
    fn=int(fn)
    tp=int(tp)
    #sys.exit(0)    
    sensitivity=round(tp/(tp+fn),2)
    #print(sensitivity)
    #sys.exit(0)    
    specificity=round(tn/(tn+fp),2)
    accuracy=round((tp + tn)/(tn + fp + fn + tp),2)
    fpr, tpr, thresholds = metrics.roc_curve(fish_list, result_list, pos_label=1)
    #auc = round(metrics.auc(fpr, tpr),2)
    #print(name , "TN:",tn, "FP:",fp, "FN:",fn, "TP:",tp, "Sensitivity:",sensitivity, "Specificity:",specificity, "AUC:",auc,"Accuracy:",accuracy)
    #print(name,tn,fp,fn,tp,sensitivity,specificity,auc,accuracy)
    target_names = ['class 0', 'class 1']
    cr = classification_report(fish_list, result_list, target_names=target_names)
    cr = re.sub('\n+', '\n', cr)
    cr = re.sub(' +', ' ', cr)
    cr = re.sub('\n ', '\n', cr)
    cr_a = cr.split("\n")
    pres = []
    recall = []
    f1 = []
    for i in range(1,3,1):
        cr_a_t = cr_a[i].split(" ")
        pres.append(cr_a_t[len(cr_a_t)-4])
        recall.append(cr_a_t[len(cr_a_t)-3])
        f1.append(cr_a_t[len(cr_a_t)-2])
    presc = ' '.join(pres)
    recallc = ' '.join(recall)
    f1c = ' '.join(recall)
    weighted_acc=cr_a[5].split(" ")[2]
    
    #test_precision, test_recall, _ = precision_recall_curve(fish_list, prob)
    #auc1 =  auc(test_recall, test_precision)
    fpr, tpr, thresholds = metrics.roc_curve(fish_list, prob)
    auc1 = round(metrics.auc(fpr, tpr),2)
    print(name,tn,fp,fn,tp,sensitivity,specificity,auc1,accuracy,presc,recallc,f1c,weighted_acc)
    #print(cr_a)



#print("name","tn","fp","fn","tp","sensitivity","specificity","auc","accuracy","presc1","presc2","recallc1","recallc2","f1_score_c1","f1_score_c2","weighted_acc")  
myfile = open(seed+"_all.txt", mode='wt')
y_pred = predictor.predict(test_df, model='NeuralNetTorch')
y_prob = predictor.predict_proba(test_df)
y_pred=[int(i) for i in list(y_pred)]
y_true=[int(i) for i in list(test_df["label"])]
#report(y_true, y_pred,list(y_prob.iloc[:,1]), "test")
for i in range(0,len(y_true)):
    inx_mx=np.argmax(y_prob.iloc[i,:])
    inx_mx_val=y_prob.iloc[i,inx_mx]
    myfile.write("autogluon"+" "+str(i)+" "+str(y_true[i])+" "+"test"+" "+str(inx_mx_val)+" "+str(inx_mx)+"\n")
y_pred = predictor.predict(val_df, model='NeuralNetTorch')
y_prob = predictor.predict_proba(val_df)
y_pred=[int(i) for i in list(y_pred)]
y_true=[int(i) for i in list(val_df["label"])]
#report(y_true, y_pred,list(y_prob.iloc[:,1]), "val")
for i in range(0,len(y_true)):
    inx_mx=np.argmax(y_prob.iloc[i,:])
    inx_mx_val=y_prob.iloc[i,inx_mx]
    myfile.write("autogluon"+" "+str(i)+" "+str(y_true[i])+" "+"val"+" "+str(inx_mx_val)+" "+str(inx_mx)+"\n")
y_pred = predictor.predict(train_df, model='NeuralNetTorch')
y_prob = predictor.predict_proba(train_df)
y_pred=[int(i) for i in list(y_pred)]
y_true=[int(i) for i in list(train_df["label"])]
#report(y_true, y_pred,list(y_prob.iloc[:,1]), "train")
for i in range(0,len(y_true)):
    inx_mx=np.argmax(y_prob.iloc[i,:])
    inx_mx_val=y_prob.iloc[i,inx_mx]
    myfile.write("autogluon"+" "+str(i)+" "+str(y_true[i])+" "+"train"+" "+str(inx_mx_val)+" "+str(inx_mx)+"\n")
myfile.close()

