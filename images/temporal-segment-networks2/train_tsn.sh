#!/usr/bin/env bash

DATASET=$1
MODALITY=$2

TOOLS=lib/caffe-action/build/install/bin
LOG_FILE=/userhome/TSN/logs/multi-${DATASET}_${MODALITY}_split1.log
N_GPU=4
MPI_BIN_DIR=/app/lib/MPI/mpich/


echo "logging to ${LOG_FILE}"

${MPI_BIN_DIR}mpirun --allow-run-as-root  -np $N_GPU \
$TOOLS/caffe train --solver=/userhome/TSN/files/${DATASET}/tsn_bn_inception_${MODALITY}_solver.prototxt  \
   --weights=models/bn_inception_${MODALITY}_init.caffemodel 2>&1 | tee ${LOG_FILE}
