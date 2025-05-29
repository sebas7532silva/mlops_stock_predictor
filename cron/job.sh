#!/bin/bash
cd /ruta/mlops_stock_predictor
source /venv311/Scripts/Activate.ps1
python train_and_promote.py >> logs/cron.log 2>&1
