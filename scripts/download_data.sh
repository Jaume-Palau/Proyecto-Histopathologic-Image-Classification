#!/bin/bash

# Verifica si ya se tiene el dataset descargado en la rauta
if [ -f data/raw/histopathologic-cancer-detection/train_labels.csv ]; then
    echo "Dataset ya descargado"
    exit 0
fi

# Crea el directorio para los datos
mkdir -p data/raw 

# Login a kaggle para poder hacer la descarga
kaggle auth login

# Descarga el zip en el directorio de data/raw
kaggle competitions download \
  -c histopathologic-cancer-detection \
  -p data/raw

# Descomprime el archivo en el mismo directorio
unzip data/raw/histopathologic-cancer-detection.zip \
  -d data/raw/histopathologic-cancer-detection

# Elimina el zip para liberar 7GB
rm data/raw/histopathologic-cancer-detection.zip