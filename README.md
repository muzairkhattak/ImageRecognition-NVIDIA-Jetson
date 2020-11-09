# Classification_models_Pytorch_2_tensorrt
Convertts pytorch models to onnx and then to tensorrt for faster inference and memory efficiency.
# Requirements

  torch==1.5.1+cu101

  torchvision==0.6.1+cu101
  
  onnx==1.4.1
  
  tensorrt==7.0.0.11
  
  pycuda==2019.1.2

# Tensorrt Debian Installation

## Procedure

####    1) Download the TensorRT local repo file that matches the Ubuntu version and CPU architecture that you are using.
       https://developer.nvidia.com/tensorrt.
####     2) Install TensorRT from the Debian local repo package.

    os="ubuntu1x04"
  
    tag="cudax.x-trt7.x.x.x-ga-yyyymmdd"
  
    sudo dpkg -i nv-tensorrt-repo-${os}-${tag}_1-1_amd64.deb
  
    sudo apt-key add /var/nv-tensorrt-repo-${tag}/7fa2af80.pub
  
    sudo apt-get update
  
    sudo apt-get install tensorrt cuda-nvrtc-x-y

####    Where x-y for cuda-nvrtc is either 10-2 or 11-0.


####    For Python 3.x:

    sudo apt-get install python3-libnvinfer-dev

####    The following additional packages will be installed:

    python3-libnvinfer

####    If you plan to use TensorRT with TensorFlow:

    sudo apt-get install uff-converter-tf

####    The graphsurgeon-tf package will also be installed with the above command.
####    3) Verify the installation.

    dpkg -l | grep TensorRT
