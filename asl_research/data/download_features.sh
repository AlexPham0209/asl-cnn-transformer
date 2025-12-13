#!/bin/bash
curl -L -o data/processed/rwth-phoenix-2014t-i3d-features-mediapipe-features.zip\
  https://www.kaggle.com/api/v1/datasets/download/rabeyaakter23/rwth-phoenix-2014t-i3d-features-mediapipe-features

unzip data/processed/rwth-phoenix-2014t-i3d-features-mediapipe-features.zip\ -d data/processed/phoenixweather2014t_i3d_mediapipe
rm data/processed/rwth-phoenix-2014t-i3d-features-mediapipe-features.zip\