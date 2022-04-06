#!/usr/bin/env bash

# Train the models with predefined set of parameters
# Transfer the trained models to container specific directory

# Uploading to bucket
# Standardize multi-class
python ./train_save_model.py ./train_save_model.yml -d DATA -sl trained_models -sb mldl-models --standardize

if [ $? -eq 0 ]; then
  # Copy the trained models
  cp -r ./trained_models/* ./aws-sagemaker/container/trained_models/
  echo "Models trained and copied to container"
else
  echo "The process failed"
fi
