# OmicsFootPrint

## Installation
```
path_package=/path/conda/tf_new_epi_package
path_conda=/path/anaconda/condabin/conda

#create conda env
$path_conda create --prefix $path_package -c conda-forge python=3.9 cudatoolkit cudnn tensorflow-gpu=2.10.0
$path_conda activate $path_package
# lib install

pip install wandb pandas==1.5.3  scikit-learn==1.2.2 matplotlib==3.5.0 Pillow==9.4.0 scikit-image shapely==1.8.0 descartes==1.1.0 

```

## Model training
```

#input param
script_dir=/path/code/OmicsFootPrint
run_dir=/path/code/OmicsFootPrint/run

#create splits
#1. split slides in to train(15%),val(15%),test(15%) according to number of patches and outcome labels (example file : $script_dir/src/epi_label_pred/train_test_val_anno.batch2.xls)
#2. create directory structure images/train/0/,images/train/1/,images/val/0/,images/val/1/,images/test/0/,images/test/1/
#3. move the created image patches to these directories according to the asigned category per slides

#running the model
SLIM_SCRIPTS=$script_dir
LOGDIR=$run_dir/model_training
mkdir -p $LOGDIR
data_files=$run_dir/images

python $SLIM_SCRIPTS/train.py  \
  -t $data_files/train \
  -v $data_files/val \
  -m EfficientNetV2\
  -o Nadam \
  -p 256 \
  -l $LOGDIR \
  -r 0.00001 \
  -L BinaryCrossentropy \
  -e 100 -b 32  -V DEBUG \
  --use-multiprocessing True --filetype images

#predict the images with the trained model

#2 class
#python $SLIM_SCRIPTS/batch_predict_image.py $data_files/val $LOGDIR/EfficientNetV2_Nadam_1e-05-BinaryCrossentropy/my_model.h5 val > $LOGDIR/val.txt
#python $SLIM_SCRIPTS/batch_predict_image.py $data_files/train $LOGDIR/EfficientNetV2_Nadam_1e-05-BinaryCrossentropy/my_model.h5 train > $LOGDIR/train.txt
#python $SLIM_SCRIPTS/batch_predict_image.py $data_files/test $LOGDIR/EfficientNetV2_Nadam_1e-05-BinaryCrossentropy/my_model.h5 test > $LOGDIR/test.txt
#cat $LOGDIR/train.txt $LOGDIR/val.txt $LOGDIR/test.txt |grep -v "loaded" > $LOGDIR/all.txt
#check model performance
#python $SLIM_SCRIPTS/metrics_patchlevel.py $LOGDIR/all.txt > $LOGDIR/metrics.txt

#morethan 2 classes
python $SLIM_SCRIPTS/batch_predict_image_multicat.py $data_files/val $LOGDIR/EfficientNetV2_Nadam_1e-05-CategoricalCrossentropy/my_model.h5 val > $LOGDIR/val.txt
python $SLIM_SCRIPTS/batch_predict_image_multicat.py $data_files/train $LOGDIR/EfficientNetV2_Nadam_1e-05-CategoricalCrossentropy/my_model.h5 train > $LOGDIR/train.txt
python $SLIM_SCRIPTS/batch_predict_image_multicat.py $data_files/test $LOGDIR/EfficientNetV2_Nadam_1e-05-CategoricalCrossentropy/my_model.h5 test > $LOGDIR/test.txt
cat $LOGDIR/train.txt $LOGDIR/val.txt $LOGDIR/test.txt|grep -v "loaded" > $LOGDIR/all.txt
#check model performance
python $SLIM_SCRIPTS/metrics_patchlevel_multicat.py $LOGDIR/all.txt > $LOGDIR/metrics.txt

```