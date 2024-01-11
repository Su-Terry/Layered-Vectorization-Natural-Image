Quant + Sam

Pull SAM model to this directory
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Required Library:
OpenCV、NumPy、Matplotlib、PyTorch和segment-anything
pip install opencv-python numpy matplotlib torch
pip install git+https://github.com/saic-mdal/saic-mdal/segment-anything


Dataset:
Put all the images(jpg, png) in "simpledata/"

Run:
Open main.py, adjust parameters, and simply run.