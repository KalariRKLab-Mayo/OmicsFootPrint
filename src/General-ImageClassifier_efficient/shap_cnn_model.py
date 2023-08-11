#!/usr/bin/env python
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import shap
from tensorflow.keras import models, backend as K
from tensorflow.keras.preprocessing import image
import argparse
# Set log level for 'numba'
import logging
from pathlib import Path
import glob
import tensorflow as tf
tf.keras.backend.clear_session()

logging.getLogger('numba').setLevel(logging.WARNING)


# Define model, input, and output paths
#input_file = sys.argv[1].strip()
#filepath = "/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/reanalysis/TCGA_BRCA_PAM50/lung/geneexpr_cnv_rppa/model.1009.Effi/EfficientNetV2_Nadam_1e-05-CategoricalCrossentropy/my_model.h5"
#input_folder = "/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/reanalysis/TCGA_BRCA_PAM50/lung/geneexpr_cnv_rppa/images.1009"
#output_folder = "/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/xiaojia_code/Kevin_TCGA/reanalysis/TCGA_BRCA_PAM50/lung/geneexpr_cnv_rppa/images.1009_test_outputsegment"
###############################################################################
# Input Arguments
###############################################################################

parser = argparse.ArgumentParser(description='Create shap output.')
parser.add_argument("-m", "--model_filepath",
                    dest='filepath',
                    required=True,
                    help="File path to model")
parser.add_argument("-i", "--input_file",
                    dest='input_file',
                    required=True,
                    help="File path to input images")
parser.add_argument("-o", "--output_folder",
                    dest='output_folder',
                    required=True,
                    help="File path to output shap files")
parser.add_argument("-c", "--classnames",
                    dest='classnames',
                    required=True,
                    help="File path to output shap files")
args = parser.parse_args()
filepath = args.filepath
input_folder = args.input_file
output_folder = args.output_folder
classnames = args.classnames

#output_folder.mkdir(parents=True, exist_ok=True)

class_names = classnames.split(",")
#class_names = ['0', '1']
Path(output_folder).mkdir(parents=True, exist_ok=True)

# Load model
new_model = models.load_model(filepath)
IMAGE_SHAPE = (256, 256)
#files = [input_file]
#files=glob.glob(input_folder+'/*/*/*.*g')
with open(input_folder) as f:
    files = f.read().splitlines()


def model_predict(x):
    tmp = x.copy()
    b = K.constant(tmp)
    b = tf.cast(b, tf.float32) / 255
    b = tf.image.resize(b, IMAGE_SHAPE)
    return new_model(b)

for imgpath in files:
    orig = imgpath
    imgpath1 = os.path.basename(imgpath)
    tmp1 = np.asarray(Image.open(orig).convert("RGB").resize(IMAGE_SHAPE))
    files_img = [tmp1]
    X = np.asarray(files_img)

    # Define a masker
    masker = shap.maskers.Image("inpaint_telea", X[0].shape)
    explainer = shap.Explainer(model_predict, masker, output_names=class_names)

    # Explain the image
    shap_values = explainer(X[0:1], max_evals=20000, batch_size=50, outputs=shap.Explanation.argsort.flip[:1])

    # Save SHAP values plot
    shap.image_plot(shap_values, show=False)
    plt.savefig(output_folder + '/' + imgpath1 + '.shap_sample20k.png')
    plt.close()

    shapval = shap_values.values[0, :, :, 0, 0]
    rw_nms = ["y" + str(i) for i in range(1, 257, 1)]
    cl_nms = ["x" + str(i) for i in range(1, 257, 1)]
    df = pd.DataFrame(shapval, index=rw_nms, columns=cl_nms)
    df.to_csv(output_folder + '/' + imgpath1 + '.shap_sample20k.txt', index=True, header=True, sep='\t')
