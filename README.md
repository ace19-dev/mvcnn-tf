# MVCNN
![](assets/mvcnn_framework.png)

## Data
- modelnet
- You can create 2D dataset from 3D objects (.obj, .stl, and .off), using BlenderPhong

## Quick Start
- prepare multi-view image
- train.py

## Evaluation
- eval.py

## Retrieval
- deep metric learning
- extract features by train.py
- prepare query/gallery sources
- retrieve.py
    
## TODO
- improve retrieval module.
- balanced sampling batch.

## Notice
- The network architecture used in our experiments is relatively shallow to 
allow for fast training and inference.

## References from
- https://github.com/WeiTang114/MVCNN-TensorFlow
- https://github.com/tensorflow/models/tree/master/official/resnet
- https://arxiv.org/abs/1812.00442
- https://github.com/ace19-dev/gvcnn
