if grep -q 10 <<<$(hostname)
then
   exppyt=/sequoia/data1/iroccosp/caffe2/caffe2/build
fi
if grep -q 11 <<<$(hostname)
then
   exppyt=/sequoia/data2/jpeyre/libs/caffe2_deploy
fi 
exppat=/usr/cuda-7.0/bin
explib=/sequoia/data2/jalayrac/src/cudnn/cudnn-6.0/lib64/:/usr/local/cuda-8.0/lib64:/sequoia/data2/jpeyre/libs/caffe2_deploy/lib:/usr/local/cuda-8.0/extras/CUPTI/lib64:/usr/local/cuda-8.0/lib64:/sequoia/data2/gpunodes_shared_libs/cudnn/5.1/lib64:/usr/local/cuda-8.0:/usr/local/cuda-7.0/lib64/
export PATH=$exppat:$PATH
export LD_LIBRARY_PATH=$explib:$LD_LIBRARY_PATH
export PYTHONPATH=/sequoia/data1/gcheron/code/detectron/lib:$exppyt:$PYTHONPATH

