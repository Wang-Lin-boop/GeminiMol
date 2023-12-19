# GeminiMol

![](../imgs/geminimol.png)  

## Source Code

```
model/:
    GeminiMol.py: 
        Code for initiating, training and evaluating Binary-Encoders.
    CrossEncoder.py: 
        Code for initiating, training and evaluating CrossEncoders.

utils/:
    chem.py:
        Essential tools for calculating molecular propteries, MCS similarities, and handling SMILES.
    dataset_split.py:
        Divide the dataset according to the molecular skeleton.
    fingerprint.py:
        The modular tool of molecular fingerprints, including molecular similarity calculation and feature extraction.

CrossEncoder_Training.py: 
    Scripts for training and testing the CrossEncoders.

GeminiMol_Training.py:
    Scripts for training and testing the GeminiMol models.

FineTuning.py:
    Fine-tuning on downstream task using a pre-trained GeminiMol model.

PropDecoder.py:
    Trianing a proptery decoder on downstream task using fixed GeminiMol encoders, fingerprints and CrossEncoders. 

AutoQSAR.py:
    Trianing a AutoGluon decoder on downstream task using fixed GeminiMol encoders, fingerprints and CrossEncoders. 

benchmark.py:
    Benchmarking a presentation method (fingerprints, GeminiMol, or CrossEncoder) on the benchmark datasets. 

Analyzer.py:
    Analysis the given molecules based on the a series encoders, inculding GeminiMol encoders, fingerprints, CrossEncoders, and their combination. Supported modes include clustering, dimensionality reduction, heatmaps, and feature visualisation.

Screener.py:
    Performing virtual screeening on given compound database using fingerprints or GeminiMol encoders.

```

## Citing this work

**Conformational Space Profile Enhances Generic Molecular Representation Learning**     
Lin Wang, Shihang Wang, Hao Yang, Shiwei Li, Xinyu Wang, Yongqi Zhou, Siyuan Tian, Lu Liu, Fang Bai    
bioRxiv 2023.12.14.571629; doi: https://doi.org/10.1101/2023.12.14.571629    


## Get in Touch

We welcome community contributions of extension tools based on the GeminiMol model, etc. If you have any questions not covered in this overview, please contact the [GeminiMol Developer Team](wanglin3@shanghaitech.edu.cn). We would love to hear your feedback and understand how GeminiMol has been useful in your research. Share your stories with us at wanglin3@shanghaitech.edu.cn or baifang@shanghaitech.edu.cn.       

In addition to GitHub, we offer a WeChat community to provide a forum for discussion between users. You can access the community's QR code by following the "蛋白矿工" on WeChat.    

## Acknowledgements

We appreciate the technical support provided by the engineers of the high-performance computing cluster of ShanghaiTech University.  Lin Wang also thanks Jianxin Duan, Gaokeng Xiao, Quanwei Yu, Zheyuan Shen, Shenghao Dong, Huiqiong Li, Zongquan Li, and Fenglei Li for providing technical support, inspiration and help for this work.      

We appreciate the developers of AutoGluon and Deep Graph Library (DGL). We also thank the developers and maintainers of MarcoModel and PhaseShape modules in the Schrödinger package.      

Besides, GeminiMol communicates with and/or references the following separate libraries and packages, we thank all their contributors and maintainers!  

*  [_RDKit_](https://www.rdkit.org/)
*  [_PyTorch_](https://pytorch.org/)
*  [_AutoGluon_](https://auto.gluon.ai/stable/index.html)
*  [_DGL-Life_](https://lifesci.dgl.ai/)
*  [_ODDT_](https://oddt.readthedocs.io/en/latest/)
*  [_SciPy_](https://scipy.org/)
*  [_scikit-learn_](https://scikit-learn.org/stable/)
*  [_matplotlib_](https://matplotlib.org/)
