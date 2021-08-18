# BNN.pytorch
Binarized Neural Network (BNN) for pytorch
This is the (forked) pytorch version for the BNN code, for VGG and resnet models
Link to the paper: https://papers.nips.cc/paper/6573-binarized-neural-networks

The code is based on https://github.com/eladhoffer/convNet.pytorch
Please install torch and torchvision by following the instructions at: http://pytorch.org/
To run resnet18 for cifar10 dataset use: python main_binary.py --model resnet_binary --save resnet18_binary --dataset cifar10

# Dependencies

Python dependencies are managed by poetry package manager. Use `poetry shell` to start virtual environment