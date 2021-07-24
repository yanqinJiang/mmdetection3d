#!/usr/bin/env bash

spring.submit run --gpu -n1 --gres=gpu:1 \
"python tools/train.py \
configs/second/custom_hv_second_secfpn_6x8_80e_kitti-3d-3class.py \
--work-dir experiments/debug"