#!/bin/bash
curl -L -o data/external/phoenixweather2014t.zip\
  https://www.kaggle.com/api/v1/datasets/download/mariusschmidtmengin/phoenixweather2014t-3rd-attempt

unzip data/external/phoenixweather2014t.zip -d data/external/phoenixweather2014t
rm data/external/phoenixweather2014t.zip