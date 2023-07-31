#!/usr/bin/env python
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
#from __future__ import absolute_import, division, print_function, unicode_literals
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
import cv2
import numpy as np
from PIL import Image
import shap
from skimage import measure
import numpy as np
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
from descartes import PolygonPatch
import glob
import matplotlib.pyplot as plt
# In[5]:
#import librosa
import logging
import pandas as pd
from pathlib import Path
#creating a new directory called pythondirectory


logging.getLogger('numba').setLevel(logging.WARNING)
#input_file=sys.argv[1]
#input_file = input_file.strip()
#print(input_file)
#sys.exit(0)
filepath='/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/reanalysis/TCGA_BRCA_PAM50/geneexpr_cnv_rppa/model.1001/EfficientNetV2_Nadam_1e-05-CategoricalCrossentropy/my_model.h5'
input_folder="/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/reanalysis/TCGA_BRCA_PAM50/geneexpr_cnv_rppa/images.1001"
output_folder="/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/reanalysis/TCGA_BRCA_PAM50/geneexpr_cnv_rppa/images.1001_test_outputsegment"

Path(output_folder).mkdir(parents=True, exist_ok=True)

class_names=['0','1','2','3']


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
new_model = models.load_model(filepath)
IMAGE_SHAPE = (256, 256) 


files=glob.glob(input_folder+'/*/*/*.*g')
def f(x):
    tmp = x.copy()
    b = K.constant(tmp)
    #b = tf.io.decode_jpeg(b, channels=3)
    b = tf.cast(b, tf.float32)/255
    #b = tf.io.decode_jpeg(b, channels=3)
    b = tf.image.resize(b, IMAGE_SHAPE)
    #b = tf.image.rgb_to_hsv(b)
    #b = tf.reshape(b, (IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3))
    #tmp=K.eval(b)
    #tmp = np.expand_dims(tmp, axis=0)
    return new_model(b)
    
num_select_bkgrnd=100
IMAGE_SHAPE = (256, 256) 
selected_indices = np.random.choice(len(files), size=num_select_bkgrnd, replace=False)
X_train = np.zeros((num_select_bkgrnd,) + (256,256,3))
# Loop through each image path and load the corresponding image into X_train
for i, idx in enumerate(selected_indices):
    # Convert image to numpy array and divide by 255
    x = np.asarray(Image.open(files[idx]).convert("RGB").resize(IMAGE_SHAPE))/ 255.0
    # Assign numpy array to the corresponding index in X_train
    X_train[i] = x
    
imgpath=files[0]
orig=imgpath #os.path.join(input_folder,imgpath)
imgpath1=os.path.basename(imgpath)
imgpath=imgpath.replace('.jpg','_out.jpg')
tmp1=np.asarray(Image.open(orig).convert("RGB").resize(IMAGE_SHAPE))
files_img=[]
files_img.append(tmp1)
X = np.asarray(files_img)/255
#X = np.expand_dims(X, axis=0)
#background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
# explain predictions of the model on four images
shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough
shap.explainers._deep.deep_tf.op_handlers["DepthwiseConv2dNative"] = shap.explainers._deep.deep_tf.passthrough
e = shap.DeepExplainer(new_model, X_train)
shap_values = e.shap_values(X)
print(shap_values.shape)