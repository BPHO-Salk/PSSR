# Point-Scanning Super-Resolution (PSSR)

## Good stuff:

BioRxiv Preprint: [Deep Learning-Based Point-Scanning Super-Resolution Imaging](https://www.biorxiv.org/content/10.1101/740548v3)

Pretrained models can be downloaded here: 
* [PSSR for Electron Microscopy (EM)](https://www.dropbox.com/s/4o8n1jc1piivohz/PSSR_EM.pkl?dl=0)
* [PSSR singleframe (PSSR-SF) for mitoTracker](https://www.dropbox.com/s/jfsze6ro6boefzt/PSSR-SF_mitotracker.pkl?dl=0)
* [PSSR multiframe (PSSR-MF) for mitoTracker](https://www.dropbox.com/s/99ct6nxgndfnv3f/PSSR-MF_mitotracker.pkl?dl=0)
* [PSSR for neuronal mitochondria](https://www.dropbox.com/s/dlj6kbnch291wmk/PSSR_neuronalMito.pkl?dl=0)

I know this is unusual, but we do have a beautifully written [PSSR Tweetorial](https://twitter.com/manorlaboratory/status/1169624396891185152?s=20). Enjoy!

## Instruction of use

### Environment set up
- Install Anaconda ([Learn more](https://docs.anaconda.com/anaconda/install/))
- Create a new conda environment for PSSR: `conda env create --file=env.yaml` ([Learn more](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file))

### EM inference
Please refer to the handy [Inference_PSSR_for_EM.ipynb](https://github.com/BPHO-Salk/PSSR/blob/master/Inference_PSSR_for_EM.ipynb). You need to modify the path for the test images accordingly. Note the input pixel size needs to be 8 nm.
