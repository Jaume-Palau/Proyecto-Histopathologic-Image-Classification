#!/bin/bash

mkdir -p histopathologic_project/{data/raw,data/processed,notebooks,scripts,src,outputs/models} && \
touch histopathologic_project/notebooks/exploracion.ipynb \
      histopathologic_project/scripts/download_data.sh \
      histopathologic_project/src/dataset.py \
      histopathologic_project/src/transforms.py \
      histopathologic_project/src/model.py \
      histopathologic_project/src/train.py \
      histopathologic_project/src/predict.py \
      histopathologic_project/src/config.py \
      histopathologic_project/outputs/submission.csv \
      histopathologic_project/requirements.txt \
      histopathologic_project/README.md