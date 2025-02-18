#The config info
# export CUDA_HOME=/usr/local/cuda-12.0
export GPU=A100
pwd_file=$(pwd)
chmod -R *
echo $pwd_file
HYPRE_HOME=${pwd_file}/AmgT_HYPRE/src
cd ${HYPRE_HOME}

if [ "$GPU" = "A100" ]
then
    #A100
    ./configure --with-cuda --with-gpu-arch='80 80' --enable-unified-memory
    export CUDA_ARCH=-gencode arch=compute_80,code=sm_80
else
    #H100
    ./configure --with-cuda --with-gpu-arch='90 90' --enable-unified-memory
    export CUDA_ARCH=-gencode arch=compute_90,code=sm_90
fi
export HYPRE_DIR=${HYPRE_HOME}/hypre
#end config

# choose the execuative verison: AmgT_FP64, AmgT_Mixed, cuSPARSE
# execuative=AmgT_FP64

#compile the AmgT_HYPRE
# cp ${pwd_file}/config_files/${execuative}.h ${HYPRE_HOME}/seq_mv/seq_mv.h
# echo ${pwd_file}/config_files/${execuative}.h
cd ${HYPRE_HOME}
make install -j 1

#compile the test file
cd ${pwd_file}/AmgT_test
make clean && make test_new
# mv test_new ${pwd_file}/AmgT_test/runnable_files/${execuative}


