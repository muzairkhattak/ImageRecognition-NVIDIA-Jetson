# Classfication Model on Jetson 
By using the performence metrics like accuracy,total inference time and FPS we will compare the Pytorch NN model of resnet50 on the Nvidia Jetson Nano and the same Pytorch model with same training weights on the Asus GL-702VM laptop in which GTX 1060 has been installed. We will see that the performence interms of speed and FPS is lower on Nano but the accruracy is exactly same on both machines.
In this repo, I will add help regarding configuring the Jetson Nano and installing the Pytorch framework.
So before using the code files and getting help from this repository, it is necessary to have a Jeston device which has been fully set-up and one must have following things installed and ready:

-->You must have a Jetson device flashed (means you have Linux running up in your Jetson).For flashing of Nano, we simply use a imagefile (download link given below) and burn it on the SD-Card using some image writing software like etcher. You can download the imagefile for jetson Nano (its basically the program called JetPack which install OS and other AI tools in the Jetson Device) from https://developer.nvidia.com/jetson-nano-sd-card-image. Then simply insert the SD-Card into Jetson Nano and you will see your Linux running up fine. For other Jetson Devices like Jetson TX2, you will need to flash it to by having ON-BOARD flashing procedure which is explained pretty good by the tutorials of JetsonHacks, link for this is https://www.youtube.com/watch?v=s1QDsa6SzuQ&t=309s .

-->After flashing, first thing to install is the pip3 tool, which will be used to install Python Pakages later on (Use this command in terminal "sudo apt-get install python3-pip" ).

-->By having your device flashed, you will have installed the linux, TensorRT and some other related libraries for CV as well (like OPENCV, PIL etc), unfortunately Pytorch and torchvision doesn't come preinstalled with flashing. So we will first need to download and install the Pytorch specifically for JETSON DEVICES ( please don't use the normal installation method that we use in normal PC's for Pytorch). We will need to follow the steps provided by Nvidia offical Developers (yes its DUSTIN :P ). Download using this link https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-7-0-now-available/72048. I downloaded the version 1.4 for Pytorch for the experiments.

# Requirements
  Pytorch
  
The basic steps that needs to be followed are given below:
Evaulate the perfomence of the pytorch model by running the test.py script on Jetson Device (do download the preprocessed dataset and weights file and put them into the relevent location so that the test.py can access the dataset and weights file.

link for the dataset: https://drive.google.com/drive/folders/1owosdYtUmP280WcrN3ohcoSb__lnvkz_?usp=sharing&fbclid=IwAR1LPiD-tPYSNhHogdfJWSWYWjBGlQyH_1jJ84L775zUfrALQXD22DWtmg4 Credits: Guinther Kovalski

Also then run the same script on your PC and then compare the performence metrics.

For using this repo with your dataset, the dataset must be in the ImageNet style format.
-----------

RESULTS/COMPARISION
-------------
The results acheived as listed below:

Jetson Nano: 
--------
FPS = 10.6

Accuracy = 98.54 %

Asus gl702VM (My laptop):
--------
FPS = 110

Accuracy = 98.54 %
