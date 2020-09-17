#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    exit
fi

base=/data/1/data/"$1"

scp -r -P1022 \
    $base/train $base/cities_splited.csv \
    $base/data_mlabel.pt \
    $base/nsample.conf \
    $base/nodes.bin \
    konevec@server1:/data/data/"$1"

if [ "$1" == "facebook" ] || [ "$1" == "twitter" ] 
  then
    scp -r -P1022 \
        $base/geography_splited.csv \
        $base/colrow.bin \
        $base/degrees.bin \
        $base/reverse_edge_map.bin \
        $base/edge_ftrs_data.bin $base/edge_ftrs_indices.bin $base/edge_ftrs_indptr.bin \
        konevec@server1:/data/data/"$1"
fi
scp -r -P1022 $base/predict $base/predict_index.bin konevec@server1:/data/data/"$1"