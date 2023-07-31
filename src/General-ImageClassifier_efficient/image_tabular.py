from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import sklearn
from sklearn.utils import resample
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
#from kerastuner.tuners import RandomSearch
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Lambda, GlobalMaxPooling2D, GlobalAveragePooling2D, Dropout, Concatenate
import random
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
global tf_image, tf_label, status
from random import shuffle, choice
import os
import logging
from PIL import Image
logger = logging.getLogger(__name__)

#from preprocess import Preprocess, format_example, format_example_tf, update_status
#from model_factory import GetModel

#functions
def format_example(image_name=None, img_size=256):
    """
    Apply any image preprocessing here
    :param image_name: the specific filename of the image
    :param img_size: size that images should be reshaped to
    :param train: whether this is for training or not

    :return: image
    """
    global status
    train = status 	
    image = tf.io.read_file(image_name)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)/255
    image = tf.image.resize(image, (img_size, img_size))
    if train is True:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.2, seed=44)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5, seed=44)
    image = tf.reshape(image, (img_size, img_size, 3))

    return image


# basic data preparation
#X = X.astype('float32')
#y = LabelEncoder().fit_transform(y)
#seed=1002
seed=sys.argv[1]
seed = int(seed.strip())

filename="./clinical_geneexpr/data_clin_esti_rescaled_geneexpr.txt"
df = read_csv(filename, sep='\t', header=(0))
indices_luad=df[df.subtype=="LUAD"].index.tolist()
indices_lusc=df[df.subtype=="LUSC"].index.tolist()

num_15_perc=int(df["subtype"].value_counts()[0]*0.15)
random.Random(seed).shuffle(indices_luad)
random.Random(seed).shuffle(indices_lusc)
test_ind = indices_luad[:num_15_perc]+indices_lusc[:num_15_perc]
val_ind = indices_luad[-num_15_perc:]+indices_lusc[-num_15_perc:]
train_ind = indices_luad[num_15_perc:-num_15_perc]+indices_lusc[num_15_perc:-num_15_perc]

df['subtype'].replace(to_replace=["LUAD","LUSC" ], value=["0","1"], inplace=True)
df['Category']= 'Extra'
df.loc[train_ind,'Category'] = 'Train'
df.loc[val_ind,'Category'] = 'Val'
df.loc[test_ind,'Category'] = 'Test'
df.to_csv("./clinical_geneexpr/data_clin_esti_rescaled_cate_seed_"+str(seed)+".txt", sep='\t', encoding='utf-8', index=False)

outcome=["subtype"]

train_x=df[df.Category=='Train']
test_x=df[df.Category=='Val']
val_x=df[df.Category=='Test']
#train_y=train_x["subtype"].to_numpy()
#val_y=val_x["subtype"].to_numpy()
#test_y=test_x["subtype"].to_numpy()
train_y=list(train_x["subtype"])
train_y_ori=train_y
len_train=len(train_y)
train_y = [ int(i) for i in train_y]
train_y = tf.one_hot(train_y, 2)

val_y=list(val_x["subtype"])
len_val=len(val_y)
val_y = [ int(i) for i in val_y]
val_y = tf.one_hot(val_y, 2)

test_y=list(test_x["subtype"])
len_test=len(test_y)
test_y = [ int(i) for i in test_y]
test_y = tf.one_hot(test_y, 2)

train_x.drop(['Category'], axis=1,inplace=True)
val_x.drop(['Category'], axis=1,inplace=True)
test_x.drop(['Category'], axis=1,inplace=True)
train_x.drop(['subtype'], axis=1,inplace=True)
val_x.drop(['subtype'], axis=1,inplace=True)
test_x.drop(['subtype'], axis=1,inplace=True)
train_x.drop(['ID'], axis=1,inplace=True)
val_x.drop(['ID'], axis=1,inplace=True)
test_x.drop(['ID'], axis=1,inplace=True)
train_x_img_list=list(train_x['IMG'])
test_x_img_list=list(test_x['IMG'])
val_x_img_list=list(val_x['IMG'])
train_x.drop(['IMG'], axis=1,inplace=True)
val_x.drop(['IMG'], axis=1,inplace=True)
test_x.drop(['IMG'], axis=1,inplace=True)

    
train_x=train_x.to_numpy()
train_x=train_x.astype('float32')
test_x=test_x.to_numpy()
test_x=test_x.astype('float32')
val_x=val_x.to_numpy()
val_x=val_x.astype('float32')
#train_y=train_y.astype(int)
#test_y=test_y.astype(int)
#val_y=val_y.astype(int)


AUTOTUNE=1000
BATCH_SIZE=64
weights='imagenet'
include_top = False
img_size=256
NUM_WORKERS=1
ini_epoch=0
use_multiprocessing=False
input_tensor = Input(shape=(img_size, img_size, 3))
img_shape = (img_size, img_size, 3)
status=True
t_path_ds = tf.data.Dataset.from_tensor_slices(train_x_img_list)
t_image_ds = t_path_ds.map(format_example, num_parallel_calls=AUTOTUNE)
t_clin_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_x, tf.float32))
t_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_y, tf.int64)) 
t_image_label_ds = tf.data.Dataset.zip(((t_image_ds,t_clin_ds),t_label_ds))
t_image_label_ds = t_image_label_ds.repeat().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
t_steps = int(len_train/BATCH_SIZE)
status=False
v_path_ds = tf.data.Dataset.from_tensor_slices(val_x_img_list)
v_image_ds = v_path_ds.map(format_example, num_parallel_calls=AUTOTUNE)
v_clin_ds = tf.data.Dataset.from_tensor_slices(tf.cast(val_x, tf.float32))
v_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(val_y, tf.int64))
v_image_label_ds = tf.data.Dataset.zip(((v_image_ds,v_clin_ds),v_label_ds))
v_image_label_ds = v_image_label_ds.repeat().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
v_steps = int(len_val/BATCH_SIZE)
# num_img=0
# for image,clin,label in v_image_label_ds:
    # print("Image shape: ", image.numpy().shape)   
    # print("Clin: ", clin.numpy().shape)
    # print("Label: ", label.numpy().shape)
    # # #print("Label: ", label.numpy())
    # num_img=num_img+1
# print(num_img)
# sys.exit(0)

#df.groupby([outcome[0], 'Category']).size()



classes = np.unique(np.concatenate((train_y, test_y), axis=0))
# train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
# test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))
# val_x = test_x.reshape((val_x.shape[0], val_x.shape[1], 1))
# num_classes = len(np.unique(train_y))
# idx = np.random.permutation(len(train_x))
# train_x = train_x[idx]
# train_y = train_y[idx]

# def make_model(input_shape):
    # input_layer = keras.layers.Input(input_shape)

    # conv1 = Dense(100, activation="relu")(input_layer)
    # conv2 = Dense(50, activation="relu")(conv1)
    # #gap = keras.layers.GlobalAveragePooling1D()(conv2)
    # gap = Flatten()(conv2)
    # output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    # return keras.models.Model(inputs=input_layer, outputs=output_layer)

   
        
# # save orignal y because later we will use binary
# true_y = test_y.astype(np.int64)
# true_y_train = train_y.astype(np.int64)
# # transform the labels from integers to one hot vectors
# enc = sklearn.preprocessing.OneHotEncoder()
# enc.fit(np.concatenate((train_y, test_y), axis=0).reshape(-1, 1))
# train_y = enc.transform(train_y.reshape(-1, 1)).toarray()
# test_y = enc.transform(test_y.reshape(-1, 1)).toarray()
# input_shape = train_x.shape[1:]



        
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    #base_model = tf.keras.applications.VGG16(weights=weights, include_top=include_top, input_tensor=input_tensor, input_shape=img_shape)
    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(weights=weights, include_top=include_top,input_tensor=input_tensor, input_shape=img_shape)
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Flatten()(x)
    #x = Dense(25, activation="relu")(x)
    #out = Dense(self.classes, activation='softmax')(x)
    base_model.trainable = False
    
    W_init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)
    W_init_fc = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.2)
    b_init = tf.keras.initializers.TruncatedNormal(mean=0.05, stddev=0.01)
    
    input_shape=train_x.shape[1:]
    input_layer = keras.layers.Input(input_shape)
    #conv1 = Dense(100, activation="relu")(input_layer)
    #conv2 = Dense(25, activation="relu")(conv1)
    #gap = keras.layers.GlobalAveragePooling1D()(conv2)
    #gap = Flatten()(conv2)
    merged = Concatenate()([x, input_layer])
    merged = Dense(64, activation="relu")(merged)
    merged = Dense(16, activation="relu")(merged)
    prediction = Dense(2, activation='softmax',bias_initializer=b_init)(merged)
    #conv_model = Model(inputs=input_tensor, outputs=out)
    #model = make_model(input_shape=train_x.shape[1:])
    final_net = Model(inputs=[input_tensor, input_layer], outputs=prediction)
    #print(final_net.summary())
    #sys.exit(0) 
    keras.utils.plot_model(final_net, show_shapes=True,to_file='clinical_geneexpr/clinical_geneexpr_net.png')
    #sys.exit(0)        
    epochs = 500
    batch_size = 64
    #hypermodel = MyHyperModel(classes=num_classes)
    callbacks = [
        keras.callbacks.ModelCheckpoint('clinical_geneexpr/clinical_geneexpr_model_seed'+str(seed)+'.h5', save_best_only=True, monitor="val_loss" ),
        #keras.callbacks.ModelCheckpoint('cp-{epoch:04d}.ckpt', verbose=1, save_weights_only=True, save_best_only=True, monitor="val_loss"),
        #keras.callbacks.ReduceLROnPlateau(
        #    monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        #),
        #keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=0.0000001),
        keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1,patience=20,mode='min',restore_best_weights=True)
    ]
    
    
    final_net.compile(
        optimizer=tf.keras.optimizers.Nadam(lr=0.0001),
        #optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')],#,tf.keras.metrics.TruePositives(name='tp'),tf.keras.metrics.TrueNegatives(name='tn'),tf.keras.metrics.FalsePositives(name='fp'),tf.keras.metrics.FalseNegatives(name='fn')],
    )
#prev_checkpoint="/home/m081429/scripts/PAD/Dicom_html_parser/cp-0045.ckpt"
#model.load_weights(prev_checkpoint,by_name=True, skip_mismatch=True)
#

#print(final_net.get_layer("dense_1").get_config())
#sys.exit(0)           
       
from sklearn.utils import class_weight 

class_weights = class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(np.array(train_y_ori)),y=np.array(train_y_ori))
class_weights = {l:c for l,c in zip(np.unique(np.array(train_y_ori)), class_weights)}      
#print(class_weights)
#sys.exit(0)
history = final_net.fit(t_image_label_ds, steps_per_epoch=t_steps, 
              epochs=600,
              callbacks=callbacks,
              validation_data=v_image_label_ds, validation_steps=v_steps,   
              class_weight=None,
              max_queue_size=1000,
              workers=NUM_WORKERS,
              use_multiprocessing=use_multiprocessing,
              shuffle=False,initial_epoch=ini_epoch
              )
              
# history = model.fit(
    # x=(),
    # y=train_y,
    # validation_data=(test_x,test_y),
    # batch_size=batch_size,
    # epochs=epochs,
    # callbacks=callbacks,
    # verbose=1,initial_epoch=0, class_weight=class_weights_dict)
    
final_net.save('clinical_geneexpr/clinical_geneexpr_model_seed_final'+str(seed)+'.h5')
metric = "accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.savefig("clinical_geneexpr/"+metric+'_'+str(seed)+'.pdf') 
#plt.show()
plt.close()


metric = "loss"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.savefig("clinical_geneexpr/"+metric+'_'+str(seed)+'.pdf') 
#plt.show()
plt.close() 
#print(history)
#print(classes)
#print(x_test.shape,x_train.shape,y_test.shape,y_train.shape)
#sys.exit(0)
