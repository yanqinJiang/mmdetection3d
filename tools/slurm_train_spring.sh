#!/usr/bin/env bash

set -x

GPUS=$1
JOB_NAME=$2
CONFIG=$3
WORK_DIR=$4
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
spring.submit arun --job-name=${JOB_NAME} \
    --gpu \
    -n${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    -x BJ-IDC1-10-10-17-24 \
    "python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --cfg-options lr=0.002 --launcher=slurm "
