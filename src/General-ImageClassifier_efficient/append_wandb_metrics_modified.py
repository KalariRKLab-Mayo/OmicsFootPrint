import sys
import glob

#get runid
#runid=glob.glob("./wandb/latest-run/run-*.wandb")[0].split('-')[-1].replace('.wandb','')



metric_file=sys.argv[1]
metric_file = metric_file.strip()

runid=sys.argv[2]
runid=runid.strip()

fobj = open(metric_file)
header=fobj.readline().strip().split("\t")
body=[]
for file in fobj:
 file = file.strip()
 p = file.split("\t")
 body.append(p)
fobj.close()

#  wandb init
import wandb
wandb.login(key="e4bd50ce1fb03b5846449e3a51dbf217b3836253")
from wandb.keras import WandbCallback

run=wandb.init(project="EpiDermisDet", entity="dennis-dl", id=runid, resume=True)
my_table = wandb.Table(columns=header, data=body)
run.log({"summery metrics": my_table})
