#!/bin/bash

python train.py \
  --project_name "dynamic-document-embedding" \          
  --report_name "training_run" \
  --num_epochs 5 \
  --learning_rate 5e-5 \
  --batch_size 16 \
  --max_length 512 \
  --datasets squad, emrqa \
  --use_dual_loss True \
  --entropy_weight 0.01 \
  --pretrained_model_path "dmis-lab/biobert-v1.1" \
  --model_save_path "./models"