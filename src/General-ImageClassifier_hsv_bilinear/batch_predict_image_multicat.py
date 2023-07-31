from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import sys
import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np
import PIL.Image as Image
import glob
from preprocess import Preprocess, format_example, format_example_tf, update_status
from model_factory import GetModel

input_file_path=sys.argv[1]
input_file_path=input_file_path.strip()
model_path=sys.argv[2]
model_path=model_path.strip()
ori_cate=sys.argv[3]
ori_cate=ori_cate.strip()
model_name=os.path.basename(os.path.dirname(model_path))
checkpoint_dir=os.path.dirname(model_path)
model_name=os.path.basename(os.path.dirname(model_path))
model_type = model_name.split("_")[0]
optimizer = model_name.split("_")[1]
loss_function = model_name.split("-")[-1]
lr = float(model_name.split("_")[-1].replace(loss_function,"")[:-1])
#print(model_type,optimizer,loss_function,lr)
#sys.exit(0)
'''Loadmodel'''
#new_model = models.load_model(model_path)

m = GetModel(model_name=model_type, img_size=256, classes=4, num_layers=None, reg_drop_out_per=None, l2_reg=None)
new_model = m.compile_model(optimizer, lr, loss_function)
#latest = tf.train.latest_checkpoint(checkpoint_dir)
#print("hi",model_path)#,checkpoint_dir,latest)
new_model.load_weights(model_path)

'''reading the file paths'''
IMAGE_SHAPE = (256, 256)

classes = sorted(os.listdir(input_file_path))
files_list=[]
labels_list=[]
for x in classes:
    class_files = os.listdir(os.path.join(input_file_path, x))
    class_files = [os.path.join(input_file_path, x, j) for j in class_files]
    class_labels = [int(x) for y in class_files]  
    files_list = files_list+class_files
    labels_list = labels_list+class_labels
update_status(False)
t_path_ds = tf.data.Dataset.from_tensor_slices(files_list)
t_image_ds = t_path_ds.map(format_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
t_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels_list, tf.int64))
t_image_label_ds = tf.data.Dataset.zip((t_image_ds, t_label_ds,t_path_ds))
train_ds = t_image_label_ds.batch(16).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    for step, (image,label,file) in enumerate(train_ds):
        #print(step,image.shape,label.shape,file.shape)
        lst_label = list(label.numpy())
        lst_file = list(file.numpy())
        result = np.asarray(new_model.predict_on_batch(image))
        lst_result = list(result)
        for i in range(0,len(lst_label)):
            inx_mx=np.argmax(lst_result[i])
            inx_mx_val=lst_result[i][inx_mx]
            print(model_name, os.path.basename(lst_file[i].decode("utf-8")),lst_label[i],ori_cate,inx_mx_val,inx_mx)
            #sys.exit(0)
            
    
