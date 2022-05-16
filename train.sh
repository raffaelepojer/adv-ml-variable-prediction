#!/bin/bash
export PYTHONPATH=.
echo 'Start training'
python -m scripts.run-experiment code_transformer/experiments/code_transformer/variable_prediction_2.yaml