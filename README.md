# Learnable Embedding Sizes for Recommender Systems
This repository contains PyTorch Implementation of ICLR 2021 paper: [*Learnable Embedding Sizes for Recommender Systems*.](https://arxiv.org/abs/2101.07577)
Please check our paper for more details about our work if you are interested. 

## Usage
Following the steps below to run our codes:

###  1. Install torchfm
`pip install torchfm`

For more information about torchfm, please see:

<https://github.com/rixwew/pytorch-fm>

###  2. Download datasets
We provide MovieLens-1M dataset in `data/ml-1m`. If you want to run PEP on Criteo and Avazu datasets,
you need to download the dataset at [Criteo](https://www.kaggle.com/c/criteo-display-ad-challenge) and [Avazu](https://www.kaggle.com/c/avazu-ctr-prediction). 

### 3. Put the data in `data/criteo` or `data/avazu`
Raw data should be stored with the following file directory:

`data/criteo/train.txt` 

`data/avazu/train`

### 4. Specify the hyper-parameters

For learning embedding sizes, the hyper-parameters are in `train_[dataset].py`

For retraining learned embedding sizes, the hyper-parameters are in `train_[dataset]_retrain.py`

### 5. Learning embedding sizes

Run `train_[dataset].py` to learn embedding sizes. Learned embedding will be saved in
`tmp/embedding/fm/[alias]/`, named as number of parameters. 

### 6. Retrain the pruned embedding

Run `train_[dataset]_retrain.py` to retrain the pruned embedding table. You need to specify what embedding table need to be retrain by hyper-parameter `retrain_emb_param`. 

## Requirements
+ Python 3
+ PyTorch 1.1.0

## Citation
If you find this repo is useful for you, please kindly cite our paper.
```
@inproceedings{liu2021learnable,
    title={Learnable Embedding Sizes for Recommender Systems},
    author={Siyi Liu and Chen Gao and Yihong Chen and Depeng Jin and Yong Li},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=vQzcqQWIS0q}
}
```

## Acknowledgment
The structure of this code is largely based on [lambda-opt](https://github.com/yihong-chen/lambda-opt).
