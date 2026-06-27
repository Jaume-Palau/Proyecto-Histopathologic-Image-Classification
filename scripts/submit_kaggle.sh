#!/bin/bash

kaggle competitions submit \
  -c histopathologic-cancer-detection \
  -f outputs/submissions/submission.csv \
  -m "submission from local project"