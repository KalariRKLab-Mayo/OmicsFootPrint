set -x
#running the model
run_dir=./sample_data
SLIM_SCRIPTS=./src/General-ImageClassifier_efficient
LOGDIR=$run_dir/model_training
mkdir -p $LOGDIR
data_files=$run_dir/sample

#python $SLIM_SCRIPTS/train.py  \
#  -t $data_files/train \
#  -v $data_files/val \
#  -m EfficientNetV2\
#  -o Nadam \
#  -p 256 \
#  -l $LOGDIR \
#  -r 0.00001 \
#  -L CategoricalCrossentropy \
#  -e 100 -b 32  -V DEBUG \
#  --use-multiprocessing True --filetype images

#Two class
#predict the images with the trained model
#python $SLIM_SCRIPTS/batch_predict_image.py $data_files/val $LOGDIR/EfficientNetV2_Nadam_1e-05-BinaryCrossentropy/my_model.h5 val > $LOGDIR/val.txt
#python $SLIM_SCRIPTS/batch_predict_image.py $data_files/train $LOGDIR/EfficientNetV2_Nadam_1e-05-BinaryCrossentropy/my_model.h5 train > $LOGDIR/train.txt
#python $SLIM_SCRIPTS/batch_predict_image.py $data_files/test $LOGDIR/EfficientNetV2_Nadam_1e-05-BinaryCrossentropy/my_model.h5 test > $LOGDIR/test.txt
#cat $LOGDIR/train.txt $LOGDIR/val.txt $LOGDIR/test.txt |grep -v "loaded" > $LOGDIR/all.txt
#check model performance
#python $SLIM_SCRIPTS/metrics_patchlevel.py $LOGDIR/all.txt > $LOGDIR/eval.xls


#more than Two classes
#predict the images with the trained model
#python $SLIM_SCRIPTS/batch_predict_image_multicat.py $data_files/val $LOGDIR/EfficientNetV2_Nadam_1e-05-CategoricalCrossentropy/my_model.h5 val > $LOGDIR/val.txt
#python $SLIM_SCRIPTS/batch_predict_image_multicat.py $data_files/train $LOGDIR/EfficientNetV2_Nadam_1e-05-CategoricalCrossentropy/my_model.h5 train > $LOGDIR/train.txt
#python $SLIM_SCRIPTS/batch_predict_image_multicat.py $data_files/test $LOGDIR/EfficientNetV2_Nadam_1e-05-CategoricalCrossentropy/my_model.h5 test > $LOGDIR/test.txt
#cat $LOGDIR/train.txt $LOGDIR/val.txt $LOGDIR/test.txt|grep -v "loaded" > $LOGDIR/all.txt
#check model performance
#python $SLIM_SCRIPTS/metrics_patchlevel_multicat.py $LOGDIR/all.txt > $LOGDIR/metrics.txt

#find  $data_files -name '*png' > $data_files/shap_input.txt

python $SLIM_SCRIPTS/shap_cnn_model.py -m $LOGDIR/EfficientNetV2_Nadam_1e-05-CategoricalCrossentropy/my_model.h5 -i $data_files/shap_input.txt -o $data_files/shap_output -c 0,1,2,3
