CASE="1-Apple-QuantBlurSAM"

cd ImageVectorViaLayerDecomposition
cd ProcessRegionSegImg/build
./ProcessRegionSegImg $CASE
cd ../..
cd ImageVectorization/build
./ImageVectorization $CASE
cd ../..

echo "The results are in ./ImageVectorViaLayerDecomposition/Data/$CASE/results"
