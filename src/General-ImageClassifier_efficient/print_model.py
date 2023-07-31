#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import sys
import glob
import re
import tensorflow as tf
tf.keras.backend.clear_session()
from callbacks import CallBacks
from model_factory import GetModel
from preprocess import Preprocess, format_example, format_example_tf, update_status
from sklearn.utils import class_weight
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import numpy as np
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from numpy.random import seed
seed(1)

tf.random.set_seed(1)
tf.keras.backend.clear_session()

m = GetModel(model_name="EfficientNetV2", img_size=256, classes=4, num_layers=None, reg_drop_out_per=None, l2_reg=None) 
model = m.compile_model("Nadam", "0.00001", "CategoricalCrossentropy")
tf.keras.utils.plot_model(model, to_file="Effi_model.png", show_shapes=True)
#tf.keras.utils.model_to_dot(
#    model,
#    show_shapes=False)#
#    show_dtype=False,
#    show_layer_names=True,
#    rankdir="TB",
#    expand_nested=False,
#    dpi=96,
#    subgraph=False,
#    layer_range=None,
#    show_layer_activations=False,
#    show_trainable=False,
#)
#!pip install pydot
#!pip install pydot
#!pip install pydotplus
#!pip install graphviz
# !conda install graphviz
# !conda install pydot
# !conda install pydotplus
