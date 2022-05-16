#!/bin/bash
echo 'python -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-1-csn-var.yaml python train'
python -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-1-csn-var.yaml python train
echo 'python -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-1-csn-var.yaml python valid'
python -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-1-csn-var.yaml python valid 

echo 'python -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-2-var.yaml python train'
python -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-2-var.yaml python train
echo 'python -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-2-var.yaml python valid'
python -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-2-var.yaml python valid