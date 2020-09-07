#!/bin/bash

wget http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl
mv faster_rcnn_from_caffe_attr.pkl frcnn/checkpoint.pkl
wget https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/images/input.jpg
mv input.jpg frcnn/test_one.jpg
