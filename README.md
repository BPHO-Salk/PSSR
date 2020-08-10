# Point-Scanning Super-Resolution (PSSR)

This repository hosts the PyTorch implementation source code for Point-Scanning Super-Resolution (PSSR), a Deep Learning-based framework that faciliates the otherwise unattainable resolution, speed and sensitivity of point-scanning imaging systems. 

BioRxiv Preprint: [Deep Learning-Based Point-Scanning Super-Resolution Imaging](https://www.biorxiv.org/content/10.1101/740548v3)

![PSSR](example_imgs/em_example.png)


- [Overview](#overview)
- [Data Availability](#data-availablity)
- [System Requirements](#system-requirements)
- [Instruction of Use](#instruction-of-use)
- [License](#license)

# Overview

# Data Availability

All data is hosted in 3DEM Dataverse: https://doi.org/10.18738/T8/YLCK5A, which includes:
* Training, testing, and validation data
* Pretrained PSSR models for each image type described in the PSSR manuscript 
* Environment set-up env.yml file that lists all prerequisite libraries.

There is also a beautifully written [PSSR Tweetorial](https://twitter.com/manorlaboratory/status/1169624396891185152?s=20) that explains the whole development story of PSSR.

# System Requirements
Final models were generated using [fast.ai v1.0.55 library](https://github.com/fastai/fastai)

# Instruction of Use

## Run PSSR from Google Colaboratory (Colab)
Google Colaboratory (Colab) version of PSSR is now ready. ([PSSR - Colab for programmers](https://github.com/BPHO-Salk/PSSR/tree/master/colab_notebooks/))

Another PSSR Colab version that orients to non-programmers is also going to be released soon. ([PSSR - Colab for non-programmers (In progress)](https://github.com/BPHO-Salk/PSSR/tree/master/colab_notebooks/))

Very few libraries need to be installed manually for PSSR Colab verion - most dependencies are preinstalled in the Colab environment. This makes the environment set-up step painless and you will be able to quickly get straight to the real fun. However, it also means some of the libraries you will be using are more recent than the ones used for the manuscript, which can be accessed from the instruction below.

## Run PSSR from the command line 

### Environment set-up
- Install Anaconda ([Learn more](https://docs.anaconda.com/anaconda/install/))
- Create a new conda environment for PSSR: `conda env create --file=env.yaml` ([Learn more](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file))

### EM inference
Please refer to the handy [Inference_PSSR_for_EM.ipynb](https://github.com/BPHO-Salk/PSSR/blob/master/Inference_PSSR_for_EM.ipynb). You need to modify the path for the test images accordingly. Note the input pixel size needs to be 8 nm.

# Citation
Please cite our work if you find this work useful for your research: 

# License
Licensed under BSD 3-Clause License.
