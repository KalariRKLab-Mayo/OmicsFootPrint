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
#from tensorflow import set_random_seed
tf.random.set_seed(1)
tf.keras.backend.clear_session()

###############################################################################
# Input Arguments
###############################################################################

parser = argparse.ArgumentParser(description='Run a Siamese Network with a triplet loss on a folder of images.')
parser.add_argument("-t", "--image_dir_train",
                    dest='image_dir_train',
                    required=True,
                    help="File path ending in folders that are to be used for model training")

parser.add_argument("-v", "--image_dir_validation",
                    dest='image_dir_validation',
                    default=None,
                    help="File path ending in folders that are to be used for model validation")

parser.add_argument("-m", "--model-name",
                    dest='model_name',
                    default='VGG16',
                    choices=[ 'custom',
                             'DenseNet121',
                             'DenseNet169',
                             'DenseNet201',
                             'InceptionResNetV2',
                             'InceptionV3',
                             'MobileNet',
                             'MobileNetV2',
                             'NASNetLarge',
                             'NASNetMobile',
                             'ResNet50',
                             'ResNet152',
                             'ResNetRS420',
                             'VGG16',
                             'VGG19',
                             'Xception',
                             'EfficientNetV2'],
                    help="Models available from tf.keras")

parser.add_argument("-o", "--optimizer-name",
                    dest='optimizer',
                    default='Adam',
                    choices=['Adadelta',
                             'Adagrad',
                             'Adam',
                             'Adamax',
                             'Ftrl',
                             'Nadam',
                             'RMSprop',
                             'SGD'],
                    help="Optimizers from tf.keras")

parser.add_argument("-p", "--patch_size",
                    dest='patch_size',
                    help="Patch size to use for training",
                    default=256, type=int)

parser.add_argument("-l", "--log_dir",
                    dest='log_dir',
                    default='log_dir',
                    help="Place to store the tensorboard logs")

parser.add_argument("-r", "--learning-rate",
                    dest='lr',
                    help="Learning rate",
                    default=0.0001, type=float)

parser.add_argument("-L", "--loss-function",
                    dest='loss_function',
                    default='BinaryCrossentropy',
                    choices=['SparseCategoricalCrossentropy',
                             'CategoricalCrossentropy',
                             'BinaryCrossentropy','Hinge'],
                    help="Loss functions from tf.keras")

parser.add_argument("-e", "--num-epochs",
                    dest='num_epochs',
                    help="Number of epochs to use for training",
                    default=10, type=int)

parser.add_argument("-b", "--batch-size",
                    dest='BATCH_SIZE',
                    help="Number of batches to use for training",
                    default=1, type=int)

parser.add_argument("-w", "--num-workers",
                    dest='NUM_WORKERS',
                    help="Number of workers to use for training",
                    default=1, type=int)

parser.add_argument("--use-multiprocessing",
                    help="Whether or not to use multiprocessing",
                    const=True, default=False, nargs='?',
                    type=bool)

parser.add_argument("-V", "--verbose",
                    dest="logLevel",
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    default="DEBUG",
                    help="Set the logging level")

parser.add_argument("-F", "--filetype",
                    dest="filetype",
                    choices=['tfrecords', 'images'],
                    default="images",
                    help="Set the logging level")

parser.add_argument("-D", "--drop_out",
                    dest="reg_drop_out_per",
                    default=None,type=float,
                    help="Regulrization drop out percent 0-1")

parser.add_argument("-R", "--l2_reg",
                    dest="l2_reg",
                    default=None,type=float,
                    help="L2 Regulrization 0-0.1")
                    
parser.add_argument("--tfrecord_image",
                    dest="tfrecord_image",
                    default="image/encoded",
                    help="Set the logging level")

parser.add_argument("--tfrecord_label",
                    dest="tfrecord_label",
                    default="null",
                    help="Set the logging level")

parser.add_argument("--train_num_layers",
                    dest="train_num_layers",
                    default=False,
                    help="Set the logging level")

parser.add_argument("--prev_checkpoint",
                    dest="prev_checkpoint",
                    default=False,
                    help="Set the logging level")


parser.add_argument("--wan_db",
                    dest="flag_wandb",
                    default=1,
                    help="wandb flag")

parser.add_argument("--wan_db_proj",
                    dest="flag_wandb_proj",
                    default="NA",
                    help="wandb flag project")

parser.add_argument("--wan_db_runname",
                    dest="flag_wandb_runname",
                    default="NA",
                    help="wandb flag runname")
                    
args = parser.parse_args()

logging.basicConfig(stream=sys.stderr, level=args.logLevel,
                    format='%(name)s (%(levelname)s): %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(args.logLevel)


from numpy.random import seed
seed(1)
#from tensorflow import set_random_seed
tf.random.set_seed(1)
tf.keras.backend.clear_session()
#print(args.flag_wandb,args.flag_wandb_proj,args.flag_wandb_runname)

flag_wandb=int(args.flag_wandb)
flag_wandb_proj=args.flag_wandb_proj
flag_wandb_runname=args.flag_wandb_runname
if flag_wandb==0:
    if args.flag_wandb_proj == "NA" or args.flag_wandb_runname == "NA":
        print("flag_wandb_proj or flag_wandb_runname cannot be NA")
        sys.exit(0)
    #  wandb init
    import wandb
    wandb.login(key="c6dada21188b1d287881b49b826e6c68af765464") #e4bd50ce1fb03b5846449e3a51dbf217b3836253")
    from wandb.keras import WandbCallback
    notes = " "
    wandb.init(project=flag_wandb_proj, entity="dennis-dl", notes=notes)
    wandb.config.max_epochs = args.num_epochs
    wandb.config.lr = args.lr
    wandb.config.batch_size = args.BATCH_SIZE
    wandb.config.patch_size =args.patch_size
    wandb.config.optimizer = args.optimizer
    wandb.config.loss = args.loss_function
    wandb.config.model_name = args.model_name
    wandb.config.num_classes = 2
    wandb.config.drop_out = args.reg_drop_out_per
    wandb.config.l2_regul = args.l2_reg
    wandb.run.name = flag_wandb_runname 

###############################################################################
# Begin priming the data generation pipeline
###############################################################################

# Get Training and Validation data
train_data = Preprocess(args.image_dir_train, args.filetype, args.tfrecord_image, args.tfrecord_label, loss_function=args.loss_function,batch_size=args.BATCH_SIZE)
#print(len(train_data.files),len(train_data.labels),train_data.files[:5],train_data.ori_labels[:5],set(train_data.ori_labels),)
#sys.exit(1)
# print(train_data.loss_function)
# print(train_data.filetype)
# print(train_data.loss_function)
# print(train_data.classes)
# print(train_data.tfrecord_image)
# print(train_data.tfrecord_label)
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(np.array(train_data.ori_labels)),
                                                 y=np.array(train_data.ori_labels))
#print(class_weights)
class_weights = {l:c for l,c in zip(np.unique(np.array(train_data.ori_labels)), class_weights)}
# sys.exit(0)
#class_weights = class_weight.compute_class_weight('balanced',np.unique(np.array(train_data.ori_labels)), np.array(train_data.ori_labels))
#print(class_weights)
#class_weights = {l:c for l,c in zip(np.unique(np.array(train_data.ori_labels)), class_weights)}
#print(class_weights)
#sys.exit(0)

logger.debug('Completed  training dataset Preprocess')

#AUTOTUNE = tf.data.experimental.AUTOTUNE
AUTOTUNE=1000

#Update status to Training for map function in the preprocess
update_status(True)

#If input datatype is tfrecords or images
if train_data.filetype!="tfrecords":
    t_path_ds = tf.data.Dataset.from_tensor_slices(train_data.files)
    t_image_ds = t_path_ds.map(format_example, num_parallel_calls=AUTOTUNE)
    t_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_data.labels, tf.int64))
    t_image_label_ds = tf.data.Dataset.zip((t_image_ds, t_label_ds))
    #train_ds = t_image_label_ds.shuffle(buffer_size=train_data.min_images).repeat()
else:
    t_path_ds = tf.data.TFRecordDataset(train_data.files)
    t_image_ds = t_path_ds.map(format_example_tf, num_parallel_calls=AUTOTUNE)
    #sys.exit(0)
    #min images variables should be update from number  of tfrecords to number of images
    num_image=0
    for image, label in t_image_ds:
        num_image=num_image+1
    #print(num_image)
    #sys.exit(0)
    train_data.min_images=num_image
    t_image_label_ds = tf.data.Dataset.zip(t_image_ds)
    #naresh:adding these additional steps to avoid shuffling on images and shuffle on imagepaths
    #t_image_ds = t_path_ds.shuffle(buffer_size=train_data.min_images).repeat()
    #train_ds = tf.data.Dataset.zip(t_image_ds)
    #train_ds = t_image_label_ds.shuffle(buffer_size=train_data.min_images).repeat()
#train_ds = t_image_label_ds.shuffle(buffer_size=train_data.min_images)
train_ds = t_image_label_ds
#train_ds = t_image_label_ds.shuffle(buffer_size=train_data.min_images).repeat()
train_ds = train_ds.repeat().batch(args.BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
#train_ds = train_ds.batch(args.BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
training_steps = int(train_data.min_images / args.BATCH_SIZE)
logger.debug('Completed Training dataset')

# for image, label in train_ds:
    # print("Image shape: ", image.numpy().shape)   
    # print("Label: ", label.numpy().shape)
    # print("Label: ", label.numpy())
    # sys.exit(0)

if args.image_dir_validation:
    # Get Validation data
    #Update status to Testing for map function in the preprocess	
    update_status(False)	 
    validation_data = Preprocess(args.image_dir_validation, args.filetype, args.tfrecord_image, args.tfrecord_label, loss_function=args.loss_function,batch_size=args.BATCH_SIZE)
    logger.debug('Completed test dataset Preprocess')
	
    if validation_data.filetype!="tfrecords":
        v_path_ds = tf.data.Dataset.from_tensor_slices(validation_data.files)
        v_image_ds = v_path_ds.map(format_example, num_parallel_calls=AUTOTUNE)
        v_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(validation_data.labels, tf.int64))
        v_image_label_ds = tf.data.Dataset.zip((v_image_ds, v_label_ds))
    else:
        v_path_ds =  tf.data.TFRecordDataset(validation_data.files)
        v_image_ds = v_path_ds.map(format_example_tf, num_parallel_calls=AUTOTUNE)
        #min images variables should be update from number  of tfrecords to number of images		
        num_image=0
        for image, label in v_image_ds:
            num_image=num_image+1
        validation_data.min_images=num_image
        v_image_label_ds = tf.data.Dataset.zip(v_image_ds)

    #validation_ds = v_image_label_ds.shuffle(buffer_size=validation_data.min_images)
    validation_ds = v_image_label_ds
    #num_img=0
    #for image, label in validation_ds:
        #print("Image shape: ", image.numpy().shape)   
        #print("Label: ", label.numpy().shape)
        #print("Label: ", label.numpy())
        #num_img=num_img+1
    #print(num_img)
    #sys.exit(0)
    #validation_ds = validation_ds.batch(args.BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_ds.repeat().batch(args.BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    
    validation_steps = int(validation_data.min_images / args.BATCH_SIZE)
    logger.debug('Completed Validation dataset')

else:
    validation_ds = None
    validation_steps = None


out_dir = os.path.join(args.log_dir, args.model_name + '_' + args.optimizer + '_' + str(args.lr) + '-' + args.loss_function)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

checkpoint_path = os.path.join(out_dir, "cp-{epoch:04d}.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)

reg_drop_out_per=None
if args.reg_drop_out_per is not None:
    reg_drop_out_per=float(args.reg_drop_out_per)


l2_reg=None
if args.l2_reg is not None:
    l2_reg=float(args.l2_reg)

	
num_layers=None
if args.train_num_layers is not None:
    num_layers=int(args.train_num_layers)

logger.debug('Mirror initialized')
GPU = True
if GPU is True:
    # This must be fixed for multi-GPU
    mirrored_strategy = tf.distribute.MirroredStrategy()
    #mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    #mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    with mirrored_strategy.scope():
        #if args.train_num_layers:
        m = GetModel(model_name=args.model_name, img_size=args.patch_size, classes=train_data.classes, num_layers=num_layers, reg_drop_out_per=reg_drop_out_per, l2_reg=l2_reg)     
        #else:
        #    m = GetModel(model_name=args.model_name, img_size=args.patch_size, classes=train_data.classes, reg_drop_out_per=reg_drop_out_per)
        #logger.debug('Model constructed')
        
        model = m.compile_model(args.optimizer, args.lr, args.loss_function)
        #print(model.summary())
        #sys.exit(0)
        #inside scope
        logger.debug('Model compiled')
        latest = tf.train.latest_checkpoint(checkpoint_dir)
 
        if not latest:
            if args.prev_checkpoint:
                model.load_weights(args.prev_checkpoint)
                logger.debug('Loading weights from '+args.prev_checkpoint)   
            model.save_weights(checkpoint_path.format(epoch=0))
            latest = tf.train.latest_checkpoint(checkpoint_dir)

        ini_epoch=int(re.findall(r'\b\d+\b', os.path.basename(latest))[0])
        
        logger.debug('Loading initialized model')
        model.load_weights(latest)
        print(latest,ini_epoch)
        #model.save(os.path.join(out_dir, 'my_model.h5'))
        #sys.exit(0)
        logger.debug('Loading weights from ' + latest)

    logger.debug('Completed loading initialized model')
    cb = CallBacks(learning_rate=args.lr, log_dir=out_dir, optimizer=args.optimizer)
    cb_list =cb.get_callbacks()
    if flag_wandb==0:
        cb_list.append(WandbCallback(save_model=False))
    logger.debug('Model image saved')#
    model.fit(train_ds, steps_per_epoch=training_steps, 
              epochs=args.num_epochs,
              callbacks=cb_list,
              validation_data=validation_ds, validation_steps=validation_steps,   
              class_weight=class_weights,
              max_queue_size=1000,
              workers=args.NUM_WORKERS,
              use_multiprocessing=args.use_multiprocessing,
              shuffle=False,initial_epoch=ini_epoch
              )
    model.save(os.path.join(out_dir, 'my_model.h5'))
else:
    #if args.train_num_layers:
    #    m = GetModel(model_name=args.model_name, img_size=args.patch_size, classes=train_data.classes,
    #                 num_layers=int(args.train_num_layers))
    #else:
    #    m = GetModel(model_name=args.model_name, img_size=args.patch_size, classes=train_data.classes)
    m = GetModel(model_name=args.model_name, img_size=args.patch_size, classes=train_data.classes, num_layers=num_layers, reg_drop_out_per=reg_drop_out_per, l2_reg=l2_reg)     
    logger.debug('Model constructed')
    
    model = m.compile_model(args.optimizer, args.lr, args.loss_function)
    logger.debug('Model compiled')
    model.save_weights(checkpoint_path.format(epoch=0))
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if not latest:
        model.save_weights(checkpoint_path.format(epoch=0))
        latest = tf.train.latest_checkpoint(checkpoint_dir)
    ini_epoch = int(re.findall(r'\b\d+\b', os.path.basename(latest))[0])
    logger.debug('Loading initialized model')
    model.load_weights(latest)
    logger.debug('Loading weights from ' + latest)
    #if args.prev_checkpoint:
        #model.load_weights(args.prev_checkpoint)
        #logger.debug('Loading weights from ' + args.prev_checkpoint)
    logger.debug('Completed loading initialized model')
    cb = CallBacks(learning_rate=args.lr, log_dir=out_dir, optimizer=args.optimizer)
    logger.debug('Model image saved')
    model.fit(train_ds,
              steps_per_epoch=training_steps,
              epochs=args.num_epochs,
              callbacks=cb.get_callbacks(),
              validation_data=validation_ds,
              validation_steps=validation_steps,
              class_weight=None,
              max_queue_size=1000,
              workers=args.NUM_WORKERS,
              use_multiprocessing=args.use_multiprocessing,
              shuffle=False,initial_epoch=ini_epoch)
    model.save(os.path.join(out_dir, 'my_model.h5'))
