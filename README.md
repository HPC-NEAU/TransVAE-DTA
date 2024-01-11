# TransVAE-DTA: Transformer and variational autoencoder network for drug-target binding affinity prediction

Computer Methods and Programs in Biomedicine,
Volume 244,
2024,
108003,
ISSN 0169-2607,
https://doi.org/10.1016/j.cmpb.2023.108003.
(https://www.sciencedirect.com/science/article/pii/S0169260723006697)

## The architecture of TransVAE-DTA
![image](https://github.com/HPC-NEAU/TransVAE-DTA/assets/101908318/f76c0cdb-70f9-418b-bcfb-55fd67109e6c)
Figure 1. (a) The architecture of Variational Autoencoder (VAE). This model consists of two main components: an encoder block and a decoder block. The encoder block is responsible for extracting meaningful latent features from the input data, while the decoder block reconstructs the original variable structure based on these features. (b) The architecture of adaptive attention pooling (AAP) module. (c) The overall architecture of TransVAE-DTA model. (d) The framework of drug encoder.  represents element-wise multiplication.
## Authors
Changjian Zhou, Zhongzheng Li, Jia Song, Wensheng Xiang
## Abstract
Recent studies have emphasized the significance of computational in silico drug-target binding affinity (DTA) prediction in the field of drug discovery and drug repurposing. However, existing DTA prediction approaches suffer from two major deficiencies that impede their progress. Firstly, while most methods primarily focus on the feature representations of drug-target binding affinity pairs, they fail to consider the long-distance relationships of proteins. Furthermore, many deep learning-based DTA predictors simply model the interaction of drug-target pairs through concatenation, which hampers the ability to enhance prediction performance.
Methods
To address these issues, this study proposes a novel framework named TransVAE-DTA, which combines the transformer and variational autoencoder (VAE). Inspired by the early success of VAEs, we aim to further investigate the feasibility of VAEs for drug structure encoding, while utilizing the transformer architecture for target feature representation. Additionally, an adaptive attention pooling (AAP) module is designed to fuse the drug and target encoded features. Notably, TransVAE-DTA is proven to maximize the lower bound of the joint likelihood of drug, target, and their DTAs.
Results
Experimental results demonstrate the superiority of TransVAE-DTA in drug-target binding affinity prediction assignments on two public Davis and KIBA datasets.
Conclusions
In this research, the developed TransVAE-DTA opens a new avenue for engineering drug-target interactions.
## Requirements
Our work is based on GPU

python 3.7
numpy == 1.21.6
pandas == 1.3.5
networkx == 2.6.3
scikit-learn == 1.0.2
rdkit == 2022.9.5
torch >= 1.2.0
torch-geometric == 2.2.0

## Installation
You can install the required dependencies with the following code.

'<conda create -n TransVAE python rdkit pytorch=1.7.1 cudatoolkit=11.0 -c pytorch -c conda-forge --yes>'
'<conda activate TransVAE>'
'<conda install pytorch torchvision cudatoolkit -c pytorch>'
'<pip install torch-scatter==latest+cu110>'
'<pip install torch-sparse==latest+cu110>'
'<pip install torch-cluster==latest+cu110>'
'<pip install torch-spline-conv==latest+cu110>' 
'<pip install torch-geometric>'


## Dataset

All datasets can be download from https://github.com/hkmztrk/DeepDTA/tree/master/data


## Using
'<python run_experiment.py>' 
