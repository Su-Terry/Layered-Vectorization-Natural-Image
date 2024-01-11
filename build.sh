#!/bin/bash

cd ImageVectorViaLayerDecomposition
cd ProcessRegionSegImg
rm -rf build
mkdir -p build && cd build
cmake ..
make
cd ../..

cd ImageVectorization
rm -rf build
mkdir -p build && cd build
cmake ..
make
cd ../..

