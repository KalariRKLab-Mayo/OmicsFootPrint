import sys
import glob

#get runid
runid=glob.glob("./wandb/latest-run/run-*.wandb")[0].split('-')[-1].replace('.wandb','')



metric_file=sys.argv[1]
metric_file = metric_file.strip()

fobj = open(metric_file)
header=fobj.readline().strip().split(' ')
header.append('Threshold')
train=fobj.readline().strip().split(' ')
train.append('0.5')
val=fobj.readline().strip().split(' ')
val.append('0.5')
test=fobj.readline().strip().split(' ')
test.append('0.5')
test21=fobj.readline().strip().split(' ')
test21.append('0.5')

threshold=fobj.readline().strip().split(' ')[-1]
train_best=fobj.readline().strip().split(' ')
train_best.append(str(threshold))
val_best=fobj.readline().strip().split(' ')
val_best.append(str(threshold))
test_best=fobj.readline().strip().split(' ')
test_best.append(str(threshold))
test_best21=fobj.readline().strip().split(' ')
test_best21.append(str(threshold))
#for file in fobj:
#    file = file.strip()
#    p = file.split("\t")
#fobj.close()

#  wandb init
import wandb
wandb.login(key="e4bd50ce1fb03b5846449e3a51dbf217b3836253")
from wandb.keras import WandbCallback

run=wandb.init(project="PAD", entity="dennis-dl", id=runid, resume=True)
my_table = wandb.Table(columns=header, data=[train,val,test,test21,train_best,val_best,test_best,test_best21])
run.log({"summery metrics": my_table})