# iSat-Seg

An enhanced software solution for semantic segmentation of broad area satellite imagery. Previously owned by avanetten, this repository is now maintained by iFlyTech. The solution exploits the capabilities of modern GPUs for training, although it will also work on a CPU (though it may be slower).

**Code Overview**

This package segments satellite imagery over large swaths of land, or sea. The example use cases include identifying roads in high resolution using SpaceNet labels. See our blog post for more details.

**Methods**

1. Installation

		Install nvidia-docker
		Build container using the command nvidia-docker build -t isat-seg path_to_isat-seg/docker
		Clone this Github repository
		Run container using the command nvidia-docker run -it -v /raid:/raid --name isat-seg_train isat-seg
		Download SpaceNet data.
		Create training masks using a modified version of the code described in our blog. Execute these scripts in a unique conda environment. The commands below create the training images, replace "train" with "test" to create testing images.
		Train a model as per instructions in the repository.
		Test on images of arbitrary size using the instructions provided in the repository.

Please have a look into our repository and contribute to developing a more efficient solution for semantic segmentation of satellite imagery.

[![Example Image](/example_ims/mask_img998.png?raw=true "Figure 1")](/example_ims/mask_img998.png)

[![Example Image](/example_ims/unet0.png?raw=true "Figure 2")](/example_ims/unet0.png)