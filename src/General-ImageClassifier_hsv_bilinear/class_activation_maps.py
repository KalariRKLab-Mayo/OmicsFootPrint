from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import logging
import os
import sys
import io
import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np
from PIL import Image
import glob
from tensorflow.keras.preprocessing import image

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from tensorflow.keras import backend as K
import random
from tensorflow.keras.utils import plot_model
import re
tf.keras.backend.clear_session()
from callbacks import CallBacks
from model_factory import GetModel
from preprocess import Preprocess, format_example, format_example_tf, update_status
#from tf_explain.callbacks.grad_cam import GradCAMCallback
#from tf_explain.core.grad_cam import GradCAM
from pathlib import Path


from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import cv2
#sys.exit(0)
#1) Compute the model output and last convolutional layer output for the image.
#2) Find the index of the winning class in the model output.
#3) Compute the gradient of the winning class with resepct to the last convolutional layer.
#3) Average this, then weigh it with the last convolutional layer (multiply them).
#4) Normalize between 0 and 1 for visualization
#5) Convert to RGB and layer it over the original image.

#filepath='/projects/shart/digital_pathology/data/Breast_CHEK2/results/logdir/chek2_level5_tf2_ResNet50_0.01_Rmsprep_alllayers_no_prev-chkp/train/ResNet50_RMSprop_0.01-BinaryCrossentropy/my_model.h5'
#filepath='/projects/shart/digital_pathology/results/General-ImageClassifier/tcga_brca1_General-ImageClassifier_2_6_2020/train/ResNet50_SGD_0.001-BinaryCrossentropy/my_model.h5'
#filepath='/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/bw.nobg.1024.cnv_exprs_allgenes_log_bilinear_topvar/VGG16_Nadam_1e-05-BinaryCrossentropy/my_model.h5'
filepath='/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/bw.nobg.1024.cnv_exprs_allgenes_log_bilinear_topvar_segment/VGG16_Nadam_1e-05-BinaryCrossentropy/my_model.h5'
#new_model = models.load_model(filepath)

'''Loadmodel'''
#new_model = models.load_model(model_path)
checkpoint_dir=os.path.dirname(filepath)
model_name=os.path.basename(os.path.dirname(filepath))
model_type = model_name.split("_")[0]
optimizer = model_name.split("_")[1]
loss_function = model_name.split("-")[-1]
lr = float(model_name.split("_")[-1].replace(loss_function,"")[:-1])
#print(model_type,optimizer,loss_function,lr)
#sys.exit(0)
'''Loadmodel'''
#new_model = models.load_model(model_path)
m = GetModel(model_name=model_type, img_size=256, classes=2, num_layers=None, reg_drop_out_per=None, l2_reg=None)
new_model = m.compile_model(optimizer, lr, loss_function)
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)
new_model.load_weights(latest)


#print(new_model.summary())

#plot_model(new_model, show_shapes=True, to_file='model.png')
#sys.exit(0)
IMAGE_SHAPE = (256, 256)

#input_folder="/projects/shart/digital_pathology/scripts/Breast_CHEK2/General-ImageClassifier_pipeline_scripts/sample_patch"
#output_folder="/projects/shart/digital_pathology/scripts/Breast_CHEK2/General-ImageClassifier_pipeline_scripts/sample_patch_cam"
#input_folder="/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/grad_cam/input"
#output_folder="/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/grad_cam/output"
input_folder="/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/grad_cam/input_cnv_segment"
output_folder="/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/grad_cam/output_segment"

files=os.listdir(input_folder)
#layer_name="conv5_block3_3_conv"
layer_name="block4_conv3"    
for imgpath in files:
    print(imgpath)
    orig=os.path.join(input_folder,imgpath)
    imgpath=imgpath.replace('.png','_out.png')
    ###example start

    #print(new_model.summary())
    ###example end
    class_index=0
    #orig=imgpath
    intensity=0.5 
    res=250
    img = image.load_img(orig, target_size=IMAGE_SHAPE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    b = K.constant(x)
    #b = tf.io.decode_png(b, channels=3)
    b = tf.cast(b, tf.float32)/255
    #b = tf.image.per_image_standardization(b)
    b = tf.image.resize(b, IMAGE_SHAPE)
    b = tf.image.rgb_to_hsv(b)
    b = tf.reshape(b, (IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3))
    x=K.eval(b)
    x = np.expand_dims(x, axis=0) 
    #print(x.shape)
    #sys.exit(0)    
    preds = new_model.predict(x)
    preds_str=str(round(preds[0][0],2))+'_'+str(round(preds[0][1],2))
    imgpath=imgpath.replace('_out.png','_'+preds_str+'_out.png')
    #print(imgpath)
    #sys.exit(0)
    with tf.GradientTape() as tape:
        last_conv_layer = new_model.get_layer(layer_name)
        iterate = tf.keras.models.Model([new_model.inputs], [new_model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(x)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
    
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap)>0.:
        heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape((32, 32))    
    img = cv2.imread(orig)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    img = heatmap * intensity + img
    img1 = heatmap * intensity
    #cv2.imwrite(os.path.join(output_folder,imgpath), cv2.resize(img, IMAGE_SHAPE))
    cv2.imwrite(os.path.join(output_folder,imgpath), img)
    cv2.imwrite(os.path.join(output_folder,imgpath+'_heatmap.png'), img1)
