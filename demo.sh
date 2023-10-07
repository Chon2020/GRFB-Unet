#!/bin/sh
echo "--------------------------------------------------------------"
echo "Creating an environment"
cd GRFB-Unet-main
conda create -n py python=3.7
conda info --envs
eval "$(conda shell.bash hook)"
conda activate py
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch
pip install opencv-python  
pip install opencv-contrib-python 
apt update
apt install libgl1-mesa-glx -y
apt-get update
apt-get install libglib2.0-dev -y
pip install matplotlib
