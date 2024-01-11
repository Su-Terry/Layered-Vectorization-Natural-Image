CASE="1-Apple-SAM"

cd ImageVectorViaLayerDecomposition
cd ProcessRegionSegImg/build
./ProcessRegionSegImg $CASE
cd ../..
cd ImageVectorization/build
./ImageVectorization $CASE
cd ../..

echo "The results are in ./ImageVectorViaLayerDecomposition/Data/$CASE/results"
