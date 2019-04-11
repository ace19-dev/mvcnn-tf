# MVCNN
![](assets/mvcnn_framework.png)

## Data
- modelnet

## Quick Start
- prepare multi-view image
- execute train.py

## Evaluation
- eval.py

## Retrieval
- prepare retrieval sources using by create_modelnet_tf_record.py
- load retrieval target

## Done
- retrieve module: nearest neighbor queries
    - Euclidean distance
    - deep cosine metric learning: <U>learn a feature representation</U>
    
## TODO

    
## Notice
- The network architecture used in our experiments is relatively shallow to 
allow for fast training and inference.

## References from
- https://github.com/WeiTang114/MVCNN-TensorFlow
- https://github.com/ace19-dev/models/tree/master/official/resnet
- https://github.com/nwojke/cosine_metric_learning