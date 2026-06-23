#!/bin/bash

# Crea el directorio para los datos
mkdir -p data/raw 

# Descarga el zip en el directorio de data/raw
kaggle competitions download \
  -c histopathologic-cancer-detection \
  -p data/raw


# Descomprime el archivo en el mismo directorio
unzip data/raw/histopathologic-cancer-detection.zip \
  -d data/raw/histopathologic-cancer-detection