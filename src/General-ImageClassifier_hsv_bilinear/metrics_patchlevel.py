import sys
import random
import os
import argparse
import sys
import pwd
import time
import subprocess
import re
import shutil
from sklearn import metrics
import re
from sklearn.metrics import roc_curve,roc_auc_score
import numpy as np
import glob
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import classification_report
import logging
import io
import tensorflow as tf
#from callbacks import CallBacks
#from model_factory import GetModel
#from preprocess import Preprocess, format_example
import matplotlib.pylab as plt
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np
from PIL import Image
import os
import re

from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc

#print(os.environ.get('CUDA_VISIBLE_DEVICES'))
#sys.exit(0)
def report1(fish_list, result_list,prob, name):
    tn, fp, fn, tp = metrics.confusion_matrix(fish_list, result_list).ravel()
    tn=int(tn)
    fp=int(fp)
    fn=int(fn)
    tp=int(tp)
    #print(name,tn, fp, fn, tp)
    # tn = len([ i for i in range(0,len(fish_list),1) if fish_list[i]==0  and result_list[i]==0] )
    # tp = len([ i for i in range(0,len(fish_list),1) if fish_list[i]==1  and result_list[i]==1] )
    # fn = len([ i for i in range(0,len(fish_list),1) if fish_list[i]==1  and result_list[i]==0] )
    # fp = len([ i for i in range(0,len(fish_list),1) if fish_list[i]==0  and result_list[i]==1] )
    #print(name,tn, fp, fn, tp)
    # sys.exit(0)    
    sensitivity=round(tp/(tp+fn),2)
    # #print(sensitivity)
    # #sys.exit(0)    
    specificity=round(tn/(tn+fp),2)
    accuracy=round((tp + tn)/(tn + fp + fn + tp),2)
    fpr, tpr, thresholds = metrics.roc_curve(fish_list, result_list, pos_label=1)
    auc = round(metrics.auc(fpr, tpr),2)
    # #print(name , "TN:",tn, "FP:",fp, "FN:",fn, "TP:",tp, "Sensitivity:",sensitivity, "Specificity:",specificity, "AUC:",auc,"Accuracy:",accuracy)
    test_precision, test_recall, _ = precision_recall_curve(fish_list, prob)
    test_f1, auc = f1_score(y_true_test, y_pred_test), auc(test_recall, test_precision)
    print(name,tn,fp,fn,tp,sensitivity,specificity,auc,accuracy)
    
def report_old(fish_list, result_list,prob, name):
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
    for i in range(1,3,1):
        cr_a_t = cr_a[i].split(" ")
        pres.append(cr_a_t[len(cr_a_t)-4])
        recall.append(cr_a_t[len(cr_a_t)-3])
    presc = ' '.join(pres)
    recallc = ' '.join(recall)
    
    test_precision, test_recall, _ = precision_recall_curve(fish_list, prob)
    auc1 =  auc(test_recall, test_precision)
    print(name,tn,fp,fn,tp,sensitivity,specificity,auc1,accuracy,presc,recallc)
    

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
    
print("name","tn","fp","fn","tp","sensitivity","specificity","auc","accuracy","presc1","presc2","recallc1","recallc2","f1_score_c1","f1_score_c2","weighted_acc")  
   

   
#print("name","tn","fp","fn","tp","sensitivity","specificity","auc1","accuracy","presc1","presc2","recallc1","recallc2")    
input_file=sys.argv[1]
input_file=input_file.strip()
train_ori=[]
train_pred=[]
test_ori=[]
test_pred=[]
val_ori=[]
val_pred=[]
train_prob=[]
val_prob=[]
test_prob=[]

fobj = open(input_file)
for line in fobj:
    line = line.strip()
    line = line.split(" ")
    #if line[3] == '1':
    #    line[3] = '0'
    #else:
    #    line[3] = '1'
    model_name = line[0]
    val=1
    if float(line[4])>0.5:
        val=0
    if line[3] == "train":
        train_ori.append(int(line[2]))
        train_pred.append(val)
        train_prob.append(float(line[5]))
    if line[3] == "test":
        test_ori.append(int(line[2]))
        test_pred.append(val)
        test_prob.append(float(line[5]))
    if line[3] == "val":
        val_ori.append(int(line[2]))
        val_pred.append(val)
        val_prob.append(float(line[5]))
fobj.close()
report(test_ori, test_pred, test_prob, model_name+" test")
report(train_ori, train_pred, train_prob, model_name+" train")
report(val_ori, val_pred, val_prob, model_name+" val") 
# model_path=sys.argv[2]
# model_path=model_path.strip()
# model_name=sys.argv[3]
# model_name=model_name.strip()

# '''Loadmodel'''
# new_model = models.load_model(model_path)

# test_dir=os.path.join(dir,"test")
# train_dir=os.path.join(dir,"train")
# val_dir=os.path.join(dir,"val")
# test_dir_files=glob.glob(test_dir+'/*/*.tfrecords')
# train_dir_files=glob.glob(train_dir+'/*/*.tfrecords')
# val_dir_files=glob.glob(val_dir+'/*/*.tfrecords')
# #print(dir,test_dir,train_dir,val_dir,len(test_dir_files),len(train_dir_files),len(val_dir_files))
# #sys.exit(0)
# IMAGE_SHAPE = (256, 256)
# original_label=[]
# predicted_label=[]
# for i in test_dir_files:
    # #print(i)
    # #sys.exit(0)
    # ori_label=os.path.basename(os.path.dirname(i))
    # raw_dataset = tf.data.TFRecordDataset(i)
    # for raw_record in raw_dataset:
        # example = tf.train.Example()
        # #print(raw_record)
        # example.ParseFromString(raw_record.numpy())
        # #result = tf.train.Example.ParseFromString(example.numpy())
        # for k,v in example.features.feature.items():
            # #print(k)
            # if k == 'image/encoded':
                # # print(k, "Skipping...")
                # stream=io.BytesIO(v.bytes_list.value[0])
                # file_out = Image.open(stream)
                # #file_out.save("sampletf.png", "png")
                
                # fileout = file_out.resize(IMAGE_SHAPE).convert('RGB')
                # #fileout = im.convert("RGB", fileout)
                # file_out=np.asarray(fileout)
                # file_out = (file_out/127.5) - 1
                # file_out = np.reshape(file_out,(1,256,256,3))
                # result = np.asarray(new_model.predict(file_out))
                # pred_label=0
                # if float(result[0][1])>0.5:
                    # pred_label=1
                # #print(ori_label,pred_label,result[0][0])
                # #sys.exit(0)
                # #print(i,pred_label)
                # original_label.append(int(ori_label))
                # predicted_label.append(pred_label)
# report(original_label, predicted_label, model_name+" test")
# #sys.exit(0)



# original_label=[]
# predicted_label=[]
# for i in train_dir_files:
    # #print(i)
    # ori_label=os.path.basename(os.path.dirname(i))
    # raw_dataset = tf.data.TFRecordDataset(i)
    # for raw_record in raw_dataset:
        # example = tf.train.Example()
        # #print(raw_record)
        # example.ParseFromString(raw_record.numpy())
        # #result = tf.train.Example.ParseFromString(example.numpy())
        # for k,v in example.features.feature.items():
            # #print(k)
            # if k == 'image/encoded':
                # # print(k, "Skipping...")
                # stream=io.BytesIO(v.bytes_list.value[0])
                # file_out = Image.open(stream)
                # #file_out.save("sampletf.png", "png")
                # fileout = file_out.resize(IMAGE_SHAPE).convert('RGB')
                # #fileout = im.convert("RGB", fileout)
                # file_out=np.asarray(fileout)
                # file_out = (file_out/127.5) - 1
                # file_out = np.reshape(file_out,(1,256,256,3))
                # result = np.asarray(new_model.predict(file_out))
                # pred_label=0
                # if float(result[0][1])>0.5:
                    # pred_label=1
                # #print(result[0][0])
                # #sys.exit(0)
                # #print(i,pred_label)
                # original_label.append(int(ori_label))
                # predicted_label.append(pred_label)
# report(original_label, predicted_label, model_name+" train")



# original_label=[]
# predicted_label=[]
# for i in val_dir_files:
    # #print(i)
    # ori_label=os.path.basename(os.path.dirname(i))
    # raw_dataset = tf.data.TFRecordDataset(i)
    # for raw_record in raw_dataset:
        # example = tf.train.Example()
        # #print(raw_record)
        # example.ParseFromString(raw_record.numpy())
        # #result = tf.train.Example.ParseFromString(example.numpy())
        # for k,v in example.features.feature.items():
            # #print(k)
            # if k == 'image/encoded':
                # # print(k, "Skipping...")
                # stream=io.BytesIO(v.bytes_list.value[0])
                # file_out = Image.open(stream)
                # #file_out.save("sampletf.png", "png")
                # fileout = file_out.resize(IMAGE_SHAPE).convert('RGB')
                # #fileout = im.convert("RGB", fileout)
                # file_out=np.asarray(fileout)
                # file_out = (file_out/127.5) - 1
                # file_out = np.reshape(file_out,(1,256,256,3))
                # result = np.asarray(new_model.predict(file_out))
                # pred_label=0
                # if float(result[0][1])>0.5:
                    # pred_label=1
                # #print(result[0][0])
                # #sys.exit(0)
                # #print(i,pred_label)
                # original_label.append(int(ori_label))
                # predicted_label.append(pred_label)
# report(original_label, predicted_label, model_name+" val")
