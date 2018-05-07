#! /bin/bash

job_list="job_list"

# PARAMS
#outname="detectron_allAnn"
outname="detectron_1key"
#outname="detectron_3key"

# DALY
vidlist="/sequoia/data2/gcheron/DALY/OF_vidlist_all.txt"
outdir="/sequoia/data2/jalayrac/nips2017weakpose/DALY_"
imdir="/sequoia/data2/gcheron/DALY/images/"
#pcfg="configs/myconfigs/e2e_faster_rcnn_R-101-FPN_daly_ftFromImageNet.yaml"
#pwei="/sequoia/data2/gcheron/detectron/imagenet/train/daly_train/generalized_rcnn/model_final.pkl"
pcfg="configs/myconfigs/e2e_faster_rcnn_R-101-FPN_daly_1key_ftFromImageNet.yaml"
pwei="/sequoia/data2/gcheron/detectron/imagenet/train/daly_train_keyframes1/generalized_rcnn/model_final.pkl"


# UCF101
#vidlist="/sequoia/data2/gcheron/UCF101/detection/OF_vidlist_all.txt"
#outdir="/sequoia/data2/jalayrac/nips2017weakpose/UCF101_"
#imdir="/sequoia/data2/gcheron/UCF101/images/"
##pcfg="configs/myconfigs/e2e_faster_rcnn_R-101-FPN_ucf101_ftFromImageNet.yaml"
##pwei="/sequoia/data2/gcheron/detectron/imagenet/train/ucf101_train/generalized_rcnn/model_final.pkl"
##pcfg="configs/myconfigs/e2e_faster_rcnn_R-101-FPN_ucf101_1key_ftFromImageNet.yaml"
##pwei="/sequoia/data2/gcheron/detectron/imagenet/train/ucf101_train_keyframes1/generalized_rcnn/model_final.pkl"
#pcfg="configs/myconfigs/e2e_faster_rcnn_R-101-FPN_ucf101_3key_ftFromImageNet.yaml"
#pwei="/sequoia/data2/gcheron/detectron/imagenet/train/ucf101_train_keyframes3/generalized_rcnn/model_final.pkl"


outdir=$outdir$outname

JOBDIR=/sequoia/data1/gcheron/code/detectron
cmd="python tools/save_detections.py --cfg $pcfg --wts $pwei"

LOGDIR=/sequoia/data2/jalayrac/nips2017weakpose/detection_logs/$outname
mkdir -p $job_list $LOGDIR

index=0
while read vid
do
   index=$(($index+1))
   vid=$(echo $vid | cut -d ' ' -f1)
   JOBNAME=detect_${index}_${vid}_$outname

   {
   echo "#$ -l mem_req=15G"
   echo "#$ -l h_vmem=400G"
   echo "#$ -j y"
   echo "#$ -o $LOGDIR"
   echo "#$ -N $JOBNAME"
   echo "#$ -q gaia.q,titan.q"
   #echo "#$ -q gaia.q"

   echo "echo \$(hostname)"

   echo "if grep -q 10 <<<\$(hostname)"
   echo "then"
   echo "   exppyt=/sequoia/data1/iroccosp/caffe2/caffe2/build"
   echo "fi"
   echo "if grep -q 11 <<<\$(hostname)"
   echo "then"
   echo "   exppyt=/sequoia/data2/jpeyre/libs/caffe2_deploy"
   echo "fi"
   echo "   exppat=/sequoia/data2/jalayrac/src/anaconda2/envs/caffe2_py2/bin:/sequoia/data2/jalayrac/src/anaconda2/bin:/usr/cuda-7.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/opt/dell/srvadmin/bin:/home/jalayrac/bin"
   echo "   explib=/sequoia/data2/jalayrac/src/cudnn/cudnn-6.0/lib64/:/usr/local/cuda-8.0/lib64:/sequoia/data2/jpeyre/libs/caffe2_deploy/lib:/usr/local/cuda-8.0/extras/CUPTI/lib64:/usr/local/cuda-8.0/lib64:/sequoia/data2/gpunodes_shared_libs/cudnn/5.1/lib64:/usr/local/cuda-8.0:/usr/local/cuda-7.0/lib64/:/cm/local/apps/cuda-driver/libs/346.59/lib64/::/usr/lib/x86_64-linux-gnu/:/usr/local/lib:/usr/lib"

   echo "export PATH=\$exppat:\$PATH"
   echo "export LD_LIBRARY_PATH=\$explib:\$LD_LIBRARY_PATH"
   echo "export PYTHONPATH=/sequoia/data1/jalayrac/src/toolboxes/detectron/lib:\$exppyt:\$PYTHONPATH"
   echo "echo \$PATH"
   echo "echo \$LD_LIBRARY_PATH"
   echo "echo \$PYTHONPATH"
   echo "cd $JOBDIR"
   echo "mkdir -p $outdir/$vid"
   echo "$cmd --output-dir $outdir/$vid $imdir/$vid"
   } > $job_list/$JOBNAME.pbs
   echo $job_list/$JOBNAME.pbs

done < "$vidlist"
