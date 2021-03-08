# NERF Pytorch
 A pytorch re-implementation of NERF

## Introduction 

This is a re-implementation of original [NeRF](https://github.com/bmild/nerf). Some functions are not included in the current implementation. Currently it only supports the 'blender' datatype. More format & training options will be added later. 

The speed is about 4-7 times faster than the original repo.

## Installation 

Install the latest version of Pytorch (>=1.6.0), then 

```
pip install torchsul imageio opencv-python matplotlib
```

## Get the data 
```
bash download_example_data.sh
```

## Run the code 

```
python train.py 
```



