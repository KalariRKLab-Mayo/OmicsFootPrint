# importing libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten, GlobalMaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras import regularizers
import logging
import sys  
  
img_width, img_height = 1024, 1024
img_size = 1024  
#train_data_dir = '/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/color_no_blood/train'
#validation_data_dir = '/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/color_no_blood/val'
train_data_dir = '/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/bw_Gemcitabine_expr_wide/train'
validation_data_dir = '/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/bw_Gemcitabine_expr_wide/val'
output_dir = '/research/bsi/projects/breast/s301449.LARdl/processing/MOLI/MOLI_code/Keras/Gemcitabine.bw.nobg.1024.exprsOnly_log_cnn_class_wide'
nb_train_samples = 12864 
nb_validation_samples = 3267
epochs = 50
batch_size = 64
num_classes = 2  
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
  
# This must be fixed for multi-GPU
mirrored_strategy = tf.distribute.MirroredStrategy()
#mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
#mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
with mirrored_strategy.scope():
    model = 'custom'
    if model == 'custom':
        model = Sequential()
        model.add(Conv2D(32, (2, 2), input_shape = input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size =(2, 2)))
          
        model.add(Conv2D(32, (2, 2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size =(2, 2)))
          
        model.add(Conv2D(64, (2, 2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size =(2, 2)))
          
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
    else:
        input_tensor = Input(shape=(img_size, img_size, 3))
        model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False, input_tensor=input_tensor, input_shape=input_shape)
        base_model = model
        x = base_model.output
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Flatten()(x)
        out = Dense(num_classes, activation='softmax')(x)
        base_model.trainable = False  
        model = Model(inputs=input_tensor, outputs=out)
    model.compile(loss ='binary_crossentropy',
                         optimizer ='rmsprop',
                       metrics =['accuracy'])

  
train_datagen = ImageDataGenerator(
                rescale = 1. / 255,
                 shear_range = 0.2,
                  zoom_range = 0.2,
            horizontal_flip = True)
  
test_datagen = ImageDataGenerator(rescale = 1. / 255)
  
train_generator = train_datagen.flow_from_directory(train_data_dir,
                              target_size =(img_width, img_height),
                     batch_size = batch_size, class_mode ='binary')
  
validation_generator = test_datagen.flow_from_directory(
                                    validation_data_dir,
                   target_size =(img_width, img_height),
          batch_size = batch_size, class_mode ='binary')
# num_img=0    
# for image, label in train_generator:
    # print("Image shape: ", image.shape)
    # #print("Image shape: ", image["anchor"].numpy().shape)
    # #print("Image shape: ", image["neg_img"].numpy().shape)
    # #print("Label: ", label.numpy().shape)
    # print("Label: ", label.shape)
    # sys.exit(0)
    # #num_img=num_img+1
# print(num_img)  
# sys.exit(0)   
model.fit_generator(train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs, validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size,workers=10, use_multiprocessing=True, verbose=1)
  
model.save_weights(output_dir+'/'+'model_saved.h5')
