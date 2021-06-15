#!/bin/bash
module load anaconda/2020.11
source activate tensorflowgpu

python speaker_id.py --cfg=cfg/SincNet_TIMIT.cfg