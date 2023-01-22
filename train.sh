#!/bin/bash

ROOT="/intraoral/stylegan2-ada-pytorch/"

RESUME="${ROOT}class-conditional/pretrained/network-snapshot-005241.pkl"
OUTDIR="${ROOT}class-conditional"
GAMMA=5
DATA="${ROOT}data/train/pair"
# COND=True
CONDITION="${ROOT}data/train/condition"
KIMG=1000
ENCODER="ViT-B/16"



python train.py --encoder $ENCODER --gamma $GAMMA --resume $RESUME \
                --outdir $OUTDIR --data $DATA --cond True --condition $CONDITION \
                --kimg $KIMG 