# OmicsFootPrint

## Installation
```
path_package=/path/conda/tf_new_epi_package
path_conda=/path/anaconda/condabin/conda

#create conda env
$path_conda create --prefix $path_package -c conda-forge python=3.9 cudatoolkit cudnn tensorflow-gpu=2.10.0 r-essentials r-base
$path_conda activate $path_package
# lib install

pip install wandb pandas==1.5.3  scikit-learn==1.2.2 matplotlib==3.5.0 Pillow==9.4.0 scikit-image shapely==1.8.0 descartes==1.1.0 shap==0.42.1 opencv_python==4.7.0.72 autogluon==0.7.0 

```

## Sample data

```
#change to script directory
cd OmicsFootPrint
#tar gunzip sample data
tar -zxvf sample_data.tar.gz

#generate circos plots
dir=`pwd`/sample_data
Rscript ./src/circos.plot.cnv.snv.exprs.generic.R $dir

cd sample_data
#create sample directory with train,val & test
bash create_train_val_test.sh
```

## Model training
```

#input param
#script_dir=/path/code/OmicsFootPrint
#run_dir=/path/code/OmicsFootPrint/run

run_dir=./sample_data
script_dir=./src/General-ImageClassifier_efficient

#create splits
#1. split slides in to train(15%),val(15%),test(15%) according to number of patches and outcome labels (example file : $script_dir/src/epi_label_pred/train_test_val_anno.batch2.xls)
#2. create directory structure images/train/0/,images/train/1/,images/val/0/,images/val/1/,images/test/0/,images/test/1/
#3. move the created image patches to these directories according to the asigned category per slides

#running the model
SLIM_SCRIPTS=$script_dir
LOGDIR=$run_dir/model_training
mkdir -p $LOGDIR
data_files=$run_dir/sample

python $SLIM_SCRIPTS/train.py  \
  -t $data_files/train \
  -v $data_files/val \
  -m EfficientNetV2\
  -o Nadam \
  -p 256 \
  -l $LOGDIR \
  -r 0.00001 \
  -L CategoricalCrossentropy \ #for two class:BinaryCrossentropy
  -e 100 -b 32  -V DEBUG \
  --use-multiprocessing True --filetype images
```
## Tensorboard
```
#visualize tensorboard
tensorboard --logdir $run_dir/model_training/EfficientNe2_Nadam_1e-05-CategoricalCrossentropy/ --host localhost --port 6060
```

## Prediction
```
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
## SHAP
```
#generate shap files
find  $data_files -name '*png' > $data_files/shap_input.txt
python $SLIM_SCRIPTS/shap_cnn_model.py -m $LOGDIR/EfficientNetV2_Nadam_1e-05-CategoricalCrossentropy/my_model.h5 -i $data_files/shap_input.txt -o $data_files/shap_output -c 0,1,2,3

```

## Other Model training : DenseNet
```

#input param

run_dir=./sample_data
script_dir=./src/General-ImageClassifier_efficient

#running the model
SLIM_SCRIPTS=$script_dir
LOGDIR=$run_dir/model_training_densenet
mkdir -p $LOGDIR
data_files=$run_dir/sample

python $SLIM_SCRIPTS/train.py  \
  -t $data_files/train \
  -v $data_files/val \
  -m DenseNet201\
  -o Nadam \
  -p 256 \
  -l $LOGDIR \
  -r 0.00001 \
  -L CategoricalCrossentropy \ #for two class:BinaryCrossentropy
  -e 100 -b 32  -V DEBUG \
  --use-multiprocessing True --filetype images
  
#predict the images with the trained model
python $SLIM_SCRIPTS/batch_predict_image_multicat.py $data_files/val $LOGDIR/DenseNet201_Nadam_1e-05-CategoricalCrossentropy/my_model.h5 val > $LOGDIR/val.txt
python $SLIM_SCRIPTS/batch_predict_image_multicat.py $data_files/train $LOGDIR/DenseNet201_Nadam_1e-05-CategoricalCrossentropy/my_model.h5 train > $LOGDIR/train.txt
python $SLIM_SCRIPTS/batch_predict_image_multicat.py $data_files/test $LOGDIR/DenseNet201_Nadam_1e-05-CategoricalCrossentropy/my_model.h5 test > $LOGDIR/test.txt
cat $LOGDIR/train.txt $LOGDIR/val.txt $LOGDIR/test.txt|grep -v "loaded" > $LOGDIR/all.txt
#check model performance
python $SLIM_SCRIPTS/metrics_patchlevel_multicat.py $LOGDIR/all.txt > $LOGDIR/metrics.txt

```

## Other Model training : Bilinear
```

#input param

run_dir=./sample_data
script_dir=./src/General-ImageClassifier_hsv_bilinear

#running the model
SLIM_SCRIPTS=$script_dir
LOGDIR=$run_dir/model_training_bilin
mkdir -p $LOGDIR
data_files=$run_dir/sample

python $SLIM_SCRIPTS/train.py \
  -m VGG16 -c 128 \
  -o Nadam \
  -p 256 \
  -r 0.0001 -e 100 -b 32 -V DEBUG --filetype images \
  -t $data_files/train \
  -v $data_files/val \
  -l $LOGDIR \
  -L CategoricalCrossentropy
  
  
#predict the images with the trained model
python $SLIM_SCRIPTS/batch_predict_image_multicat.py $data_files/val $LOGDIR/VGG16_Nadam_1e-05-CategoricalCrossentropy/my_model.h5 val > $LOGDIR/val.txt
python $SLIM_SCRIPTS/batch_predict_image_multicat.py $data_files/train $LOGDIR/VGG16_Nadam_1e-05-CategoricalCrossentropy/my_model.h5 train > $LOGDIR/train.txt
python $SLIM_SCRIPTS/batch_predict_image_multicat.py $data_files/test $LOGDIR/VGG16_Nadam_1e-05-CategoricalCrossentropy/my_model.h5 test > $LOGDIR/test.txt
cat $LOGDIR/train.txt $LOGDIR/val.txt $LOGDIR/test.txt|grep -v "loaded" > $LOGDIR/all.txt
#check model performance
python $SLIM_SCRIPTS/metrics_patchlevel_multicat.py $LOGDIR/all.txt > $LOGDIR/metrics.txt

```

## Other Model training : VGG+autogluon
```

#input param

run_dir=./sample_data
script=./src/vgg_autogluon.py


python $script

#change paths in the script
#path to save autogluon model
save_path="./sample_data/vgg_ag"
# Loading Images
circos_images_full = glob.glob("./sample_data/sample/*/*/im.*.png")

```