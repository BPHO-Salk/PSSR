# Point-Scanning Super-Resolution (PSSR)

This repository hosts the PyTorch implementation source code for Point-Scanning Super-Resolution (PSSR), a Deep Learning-based framework that faciliates otherwise unattainable resolution, speed and sensitivity of point-scanning imaging systems (e.g. scanning electron or laser scanning confocal microscopes). 

BioRxiv Preprint: [Deep Learning-Based Point-Scanning Super-Resolution Imaging](https://www.biorxiv.org/content/10.1101/740548v3)

There is also a beautifully written [PSSR Tweetorial](https://twitter.com/manorlaboratory/status/1169624396891185152?s=20) that explains the whole development story of PSSR.


![PSSR](example_imgs/em_test_results.png)

- [Overview](#overview)
- [Data Availability](#data-availablity)
- [Instruction of Use](#instruction-of-use)
- [License](#license)

# Overview
Point-scanning imaging systems (e.g. scanning electron or laser scanning confocal microscopes) are perhaps the most widely used tools for high-resolution cellular and tissue imaging, benefitting from an ability to use arbitrary pixel sizes. Like all other imaging modalities, the resolution, speed, sample preservation, and signal-to-noise ratio (SNR) of point-scanning systems are difficult to optimize simultaneously. In particular, point-scanning systems are uniquely constrained by an inverse relationship between imaging speed and pixel resolution. 

Here we show these limitations can be mitigated via the use of Deep Learning-based super-sampling of undersampled images acquired on a point-scanning system, which we termed point-scanning super-resolution (PSSR) imaging. Our proposed PSSR model coule restore undersampled images by increasing optical and pixel resolution, and denoising simultaneously. The model training requires no manually acquired image pairs thanks to the "crappification" method we developed. In addition, the multi-frame PSSR apporach enabled otherwise impossible high spatiotemporal resolution fluorescence timelapse imaging.

# Data Availability

All data is hosted in 3DEM Dataverse: https://doi.org/10.18738/T8/YLCK5A, which includes:
* Training, testing, and validation data
* Pretrained PSSR models for each image type described in the PSSR manuscript 
* Environment set-up env.yml file that lists all prerequisite libraries.

# Instruction of Use

## Run PSSR from Google Colaboratory (Colab)
Google Colaboratory (Colab) version of PSSR is now ready. ([PSSR - Colab for programmers](https://github.com/BPHO-Salk/PSSR/tree/master/colab_notebooks/))

Another PSSR Colab version that orients to non-programmers is also going to be released soon. ([PSSR - Colab for non-programmers (In progress)](https://github.com/BPHO-Salk/PSSR/tree/master/colab_notebooks/))

Very few libraries need to be installed manually for PSSR Colab verion - most dependencies are preinstalled in the Colab environment. This makes the environment set-up step painless and you will be able to quickly get straight to the real fun. However, it also means some of the libraries you will be using are more recent than the ones used for the manuscript, which can be accessed from the instruction below.

## Run PSSR from the command line 

### System Requirements

#### OS Requirements
This software is only supported for Linux, and has been tested on Ubuntu 18.04.

#### Python Dependencies
PSSR is mainly written with Fastai, and final models used in the manuscript were generated using fast.ai v1.0.55 library.

### Environment Set-up
- Install Anaconda ([Learn more](https://docs.anaconda.com/anaconda/install/))
- Create a new conda environment for PSSR: `conda env create --file=env.yaml` ([Learn more](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file))

### Example Use: EM Inference
Please refer to the handy [Inference_PSSR_for_EM.ipynb](https://github.com/BPHO-Salk/PSSR/blob/master/Inference_PSSR_for_EM.ipynb). You need to modify the path for the test images accordingly. Note the input pixel size needs to be 8 nm.

# Citation
Please cite our work if you find it useful for your research: 
```
@article{Fang2019,
abstract = {Point scanning imaging systems (e.g. scanning electron or laser scanning confocal microscopes) are perhaps the most widely used tools for high resolution cellular and tissue imaging. Like all other imaging modalities, the resolution, speed, sample preservation, and signal-to-noise ratio (SNR) of point scanning systems are difficult to optimize simultaneously. In particular, point scanning systems are uniquely constrained by an inverse relationship between imaging speed and pixel resolution. Here we show these limitations can be mitigated via the use of deep learning-based super-sampling of undersampled images acquired on a point-scanning system, which we termed point-scanning super-resolution (PSSR) imaging. Oversampled, high SNR ground truth images acquired on scanning electron or Airyscan laser scanning confocal microscopes were ‘crappified' to generate semi-synthetic training data for PSSR models that were then used to restore real-world undersampled images. Remarkably, our EM PSSR model could restore undersampled images acquired with different optics, detectors, samples, or sample preparation methods in other labs. PSSR enabled previously unattainable 2nm resolution images with our serial block face scanning electron microscope system. For fluorescence, we show that undersampled confocal images combined with a multiframe PSSR model trained on Airyscan timelapses facilitates Airyscan-equivalent spatial resolution and SNR with {\~{}}100x lower laser dose and 16x higher frame rates than corresponding high-resolution acquisitions. In conclusion, PSSR facilitates point-scanning image acquisition with otherwise unattainable resolution, speed, and sensitivity.},
author = {Fang, Linjing and Monroe, Fred and Novak, Sammy Weiser and Kirk, Lyndsey and Schiavon, Cara R and Yu, Seungyoon B and Zhang, Tong and Wu, Melissa and Kastner, Kyle and Kubota, Yoshiyuki and Zhang, Zhao and Pekkurnaz, Gulcin and Mendenhall, John and Harris, Kristen and Howard, Jeremy and Manor, Uri},
doi = {10.1101/740548},
journal = {bioRxiv},
month = {jan},
pages = {740548},
title = {Deep Learning-Based Point-Scanning Super-Resolution Imaging},
url = {http://biorxiv.org/content/early/2019/09/04/740548.abstract},
year = {2019}
}
```

# License
Licensed under BSD 3-Clause License.
