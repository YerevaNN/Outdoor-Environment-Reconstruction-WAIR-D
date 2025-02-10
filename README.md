# Outdoor-Environment-Reconstruction-WAIR-D
Repository of the paper 
> [Outdoor Environment Reconstruction with Deep Learning on Radio Propagation Paths](https://arxiv.org/abs/2402.17336)
> Hrant Khachatrian, Rafayel Mkrtchyan, Theofanis P. Raptis

## Abstract

Conventional methods for outdoor environment reconstruction rely predominantly on vision-based techniques like photogrammetry and LiDAR, facing limitations such as constrained coverage, susceptibility to environmental conditions, and high computational and energy demands. These challenges are particularly pronounced in applications like augmented reality navigation, especially when integrated with wearable devices featuring constrained computational resources and energy budgets. In response, this paper proposes a novel approach harnessing ambient wireless signals for outdoor environment reconstruction. By analyzing radio frequency (RF) data, the paper aims to deduce the environmental characteristics and digitally reconstruct the outdoor surroundings. Investigating the efficacy of selected deep learning (DL) techniques on the synthetic RF dataset WAIR-D, the study endeavors to address the research gap in this domain. Two DL-driven approaches are evaluated (convolutional U-Net and CLIP+ based on vision transformers), with performance assessed using metrics like intersection-over-union (IoU), Hausdorff distance, and Chamfer distance. The results demonstrate promising performance of the RF-based reconstruction method, paving the way towards lightweight and scalable reconstruction solutions. 
     
## Project setup

1. Duplicate the `.env.sample` file and rename the copy as `.env`.
2. Download the WAIR-D dataset from the following link: https://www.mobileai-dataset.com/html/default/yingwen/DateSet/1590994253188792322.html?index=1
3. Open the `.env` file and insert the path of the downloaded dataset as the value for `RAW_DATA_DIR`.
4. Create a new conda environment using the command:
   ```commandline
   conda env create 
   ```
5. Activate the newly created conda environment with the command:
   ```commandline
   conda activate wair_d_rec
   ```
