# Point-Scanning Super-Resolution (PSSR)

## Good stuff:

BioRxiv Preprint: [Deep Learning-Based Point-Scanning Super-Resolution Imaging](https://www.biorxiv.org/content/10.1101/740548v3)

All data is hosted in 3DEM Dataverse: https://doi.org/10.18738/T8/YLCK5A, which includes:
* Training, testing, and validation data
* Pretrained PSSR models for each image type described in the PSSR manuscript 
* Environment set-up env.yml file that lists all prerequisite libraries.

There is also a beautifully written [PSSR Tweetorial](https://twitter.com/manorlaboratory/status/1169624396891185152?s=20). Enjoy!

## Instruction of use

### Environment set-up
- Install Anaconda ([Learn more](https://docs.anaconda.com/anaconda/install/))
- Create a new conda environment for PSSR: `conda env create --file=env.yaml` ([Learn more](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file))

### EM inference
Please refer to the handy [Inference_PSSR_for_EM.ipynb](https://github.com/BPHO-Salk/PSSR/blob/master/Inference_PSSR_for_EM.ipynb). You need to modify the path for the test images accordingly. Note the input pixel size needs to be 8 nm.
