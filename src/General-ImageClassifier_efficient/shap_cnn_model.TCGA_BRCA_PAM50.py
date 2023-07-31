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

logging.getLogger('numba').setLevel(logging.WARNING)

#input_folder="/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/grad_cam1/input_cnv_segment"
#filepath='/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/reanalysis/TCGA_BRCA_PAM50/geneexpr_only/model.1003/EfficientNetV2_Nadam_1e-05-CategoricalCrossentropy/my_model.h5'
#input_folder="/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/reanalysis/TCGA_BRCA_PAM50/geneexpr_only/images.1003/test"
#output_folder="/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/reanalysis/TCGA_BRCA_PAM50/geneexpr_only/images.1003_test_outputsegment"
# filepath='/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/reanalysis/TCGA_BRCA_PAM50/geneexpr_cnv/model.1008/EfficientNetV2_Nadam_1e-05-CategoricalCrossentropy/my_model.h5'
# input_folder="/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/reanalysis/TCGA_BRCA_PAM50/geneexpr_cnv/images.1008/test"
# output_folder="/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/reanalysis/TCGA_BRCA_PAM50/geneexpr_cnv/images.1008_test_outputsegment"
# filepath='/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/reanalysis/TCGA_BRCA_PAM50/geneexpr_cnv/model.1008/EfficientNetV2_Nadam_1e-05-CategoricalCrossentropy/my_model.h5'
# input_folder="/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/reanalysis/TCGA_BRCA_PAM50/geneexpr_cnv/images.1008/test"
# output_folder="/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/reanalysis/TCGA_BRCA_PAM50/geneexpr_cnv/images.1008_test_outputsegment"
#filepath='/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/reanalysis/TCGA_BRCA_PAM50/cnv_only/model.1001/EfficientNetV2_Nadam_1e-05-CategoricalCrossentropy/my_model.h5'
#input_folder="/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/reanalysis/TCGA_BRCA_PAM50/cnv_only/images.1001/test"
#output_folder="/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/reanalysis/TCGA_BRCA_PAM50/cnv_only/images.1001_test_outputsegment"
filepath='/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/reanalysis/TCGA_BRCA_PAM50/geneexpr_cnv_rppa/model.1001/EfficientNetV2_Nadam_1e-05-CategoricalCrossentropy/my_model.h5'
input_folder="/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/reanalysis/TCGA_BRCA_PAM50/geneexpr_cnv_rppa/images.1001/test"
output_folder="/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/reanalysis/TCGA_BRCA_PAM50/geneexpr_cnv_rppa/images.1001_test_outputsegment"

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
# sys.exit(0)
# m = GetModel(model_name=model_type, img_size=256, classes=2, num_layers=None, reg_drop_out_per=None, l2_reg=None)
# new_model = m.compile_model(optimizer, lr, loss_function)
# latest = tf.train.latest_checkpoint(checkpoint_dir)
# new_model.load_weights(latest)

IMAGE_SHAPE = (256, 256) 

#files=glob.glob(input_folder+'/*.*g')
files=glob.glob(input_folder+'/*/*.*g')
#len(files),files[:5]
#print(len(files))
#files_img=[]


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
    #return b

for imgpath in files:
    orig=imgpath #os.path.join(input_folder,imgpath)
    imgpath1=os.path.basename(imgpath)
    imgpath=imgpath.replace('.jpg','_out.jpg')
    tmp1=np.asarray(Image.open(orig).convert("RGB").resize(IMAGE_SHAPE))
    files_img=[]
    files_img.append(tmp1)
    

    X = np.asarray(files_img)   
    # define a masker that is used to mask out partitions of the input image.
    masker = shap.maskers.Image("inpaint_telea", X[0].shape)
    
    # create an explainer with model and image masker
    explainer = shap.Explainer(f, masker, output_names=class_names)

    # here we explain two images using 500 evaluations of the underlying model to estimate the SHAP values
    shap_values = explainer(X[0:1], max_evals=1000, batch_size=50, outputs=shap.Explanation.argsort.flip[:1])

    shap.image_plot(shap_values,show=False)
    plt.savefig(output_folder+'/'+imgpath1+'.shap_sample1000.png')
    plt.close()
    #sys.exit(0)
    fobj=open(output_folder+'/'+imgpath1+'.shap_sample1000.txt','w')
    # In[298]:


    #shap.image_plot(shap_values)
    #shap_values[0:1].shape
    #shap_values.shape


    # In[299]:


    #for i in range(4):
    #i=0
    shapval = shap_values.values[0,:,:,0,0]
    shap_values1=shapval.copy()
    shap_values2=shapval.copy()
    shap_values1[shap_values1 < 0] = 0
    shap_values2[shap_values2 > 0] = 0
    shap_values2=np.abs(shap_values2)
    shap_values1 = (shap_values1 - shap_values1.min()) / (shap_values1.max() - shap_values1.min())
    shap_values2 = (shap_values2 - shap_values2.min()) / (shap_values2.max() - shap_values2.min())
    #print(i,np.min(shapval),np.max(shapval),np.min(shap_values1),np.max(shap_values1),np.min(shap_values2),np.max(shap_values2))
    # Set the threshold value
    threshold = 0.99
    #threshold = 0.00001
    # Find the contours in the image
    contours1 = measure.find_contours(shap_values1, threshold)
    # Create a list of Polygon objects from the contours
    polygons1 = [Polygon(contour) for contour in contours1]
    # Create a MultiPolygon object from the list of Polygon objects
    multipolygon1 = MultiPolygon(polygons1)
    # Find the contours in the image
    contours2 = measure.find_contours(shap_values2, threshold)
    # Create a list of Polygon objects from the contours
    polygons2 = [Polygon(contour) for contour in contours2]
    # Create a MultiPolygon object from the list of Polygon objects
    multipolygon2 = MultiPolygon(polygons2)
    #print(len(multipolygon1),len(multipolygon2))
    print("positive\tx_min\tx_max\ty_min\ty_max\tarea\n")
    #mp1=[]
    for p1 in multipolygon1:
        np1=np.array(p1.exterior.coords) 
        y_min=np.min(np1[:,0])
        y_max=np.max(np1[:,0])
        x_min=np.min(np1[:,1])
        x_max=np.max(np1[:,1])
        print("positive",x_min,x_max,y_min,y_max,int((x_max-x_min)*(y_max-y_min)))
        fobj.write("positive"+"\t"+str(x_min)+"\t"+str(x_max)+"\t"+str(y_min)+"\t"+str(y_max)+"\t"+str(int((x_max-x_min)*(y_max-y_min)))+"\n")
        #mp1.append(Polygon([[x_min,y_min],[x_min,y_max],[x_max,y_max],[x_max,y_min],[x_min,y_min]]))
    #mp2=[]
    for p1 in multipolygon2:
        np1=np.array(p1.exterior.coords) 
        y_min=np.min(np1[:,0])
        y_max=np.max(np1[:,0])
        x_min=np.min(np1[:,1])
        x_max=np.max(np1[:,1])
        print("negative",x_min,x_max,y_min,y_max,int((x_max-x_min)*(y_max-y_min)))
        fobj.write("negative"+"\t"+str(x_min)+"\t"+str(x_max)+"\t"+str(y_min)+"\t"+str(y_max)+"\t"+str(int((x_max-x_min)*(y_max-y_min)))+"\n")
        #mp2.append(Polygon([[x_min,y_min],[x_min,y_max],[x_max,y_max],[x_max,y_min],[x_min,y_min]])) 
    fobj.close()    
    # # Overlay the polygons on the image
    # fig, ax = plt.subplots()
    # ax.imshow(X[i])
    # ax.set_aspect('equal')
    # for polygon in mp1:
        # patch = PolygonPatch(polygon, facecolor='none', edgecolor='red')
        # ax.add_patch(patch)
    # for polygon in mp2:
        # patch = PolygonPatch(polygon, facecolor='none', edgecolor='blue')
        # ax.add_patch(patch)
    # plt.show()
    #sys.exit(0)



  

