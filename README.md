# Layered-Vectorization-and-Editable-Representation-of-Natural-Images-for-Enhanced-Graphic-Editing

![image](https://github.com/user-attachments/assets/5800a055-4b7b-4dcd-94ce-69cfe9ea451d)

## Download
Initial Download
```
git clone --recursive https://github.com/Su-Terry/Layered-Vectorization-Natural-Image.git
```

Install OpenCV, nlopt, eigen3
```
sudo apt install libopencv-dev libnlopt-dev libeigen3-dev
sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen
```

## Building Environment
```
bash ./build.sh
```

## Testing
For generating layered vectorized images.
```
bash ./dev_scripts/gen_apple_SAM.sh
```
```
bash ./dev_scripts/gen_apple_Quant_SAM.sh
```
```
bash ./dev_scripts/gen_apple_Quant_Blur_SAM.sh
```

## Re-generating segmentation mask
plz check Sam+Quant folder for more details.
