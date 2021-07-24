#!/usr/bin/env bash

set -x

GPUS=$1
JOB_NAME=$2
CONFIG=$3
CHECKPOINT=$4
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
spring.submit run \
    --job-name=${JOB_NAME} \
    --gpu \
    -n${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    "python -u tools/test.py ${CONFIG} ${CHECKPOINT} --eval mAP --launcher=slurm "
