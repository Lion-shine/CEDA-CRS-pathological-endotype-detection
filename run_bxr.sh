#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29510 train.py --output_dir=bxr_savePath_.pth --eos_coef=0.8 --dataset=bxr --num_classes=4 --num_workers=4 --start_eval=50 --epochs=150 --batch_size=4
