![](.image/geminimol.png)

# GeminiMol
This repository provides the official implementation of the GeminiMol model, training data, and scripts.  

We also provide:   

1.  Scripts for data collection and analysis.    
2.  A benchmark script and datasets for virtual screening, target identification, and QSAR (drug-target binding affinity, cellar activity, ADME, and toxicity).   
3.  Benchmark results of molecular fingerprints and GeminiMol models.   

Please also refer to our paper for a detailed description of GeminiMol.

## Motivation  

_The molecular representation model is an emerging artificial intelligence technology for extracting features of small molecules. It has been widely applied in drug discovery scenarios, such as virtual screening, Quantitative Structure-Activity Relationship (QSAR) analysis, and molecular optimization._   
  
_In previous work, molecular representation models were mostly trained on the static structure of molecules, however, the small molecules in solution are highly dynamic, and their flexible conformational changes endow them with the potential to bind to drug targets. Therefore, introducing information on small molecule conformational space into molecular representation models is a promising aim. In this work, a training strategy, named GeminiMol, was proposed to **incorporate the comprehension of conformational space into the molecular representation model**._   

_The similarity between small molecules provides the opportunity for contrastive learning, as illustrated in followed figure, the shape similarity contained more pharmacological and physical information than the 2D structure and fingerprint similarity, therefore, introducing the molecular shape similarity in contrastive learning is a promising strategy._

![similarity](.image/similarity.png)

## Application

_As a potent molecular representation model, GeminiMol finds applications in **ligand-based virtual screening, target identification, and quantitative structure-activity relationship (QSAR)** modeling of small molecular drugs. Moreover, by exploring the encoding space of GeminiMol, it enables **scaffold hopping** and facilitates the generation of innovative molecules._   

![benchmark](.image/benchmark.png)


## Installation

### GeminiMol models

### Benchmark and Baseline Fingerprint Methods


## Running GeminiMol

### Virtual Screening 

### Target Identification

### QSAR

### Molecular Generation


## Citing this work


## Acknowledgements

_We appreciate the technical support provided by the engineers of the high-performance computing cluster of ShanghaiTech University. Lin Wang also thanks Jianxin Duan, Gaokeng Xiao, Quanwei Yu, Shiwei Li and Fenglei Li for providing technical support, inspiration and help for this work. We appreciate the developers of AutoGluon and Deep Graph Library (DGL), and we thank for the RetNet implementations provided by Jamie Stirling and Frank Odom. We also thank the developers and maintainers of MarcoModel and PhaseShape modules in the Schrödinger package. Besides, GeminiMol communicates with and/or references the following separate libraries and packages, we thank all their contributors and maintainers!_  

*  [RDKit](https://www.rdkit.org/)
*  [PyTorch](https://pytorch.org/)
*  [AutoGluon](https://auto.gluon.ai/stable/index.html)
*  [DGL-Life](https://lifesci.dgl.ai/)
*  [ODDT](https://oddt.readthedocs.io/en/latest/)
*  [SciPy](https://scipy.org/)
*  [scikit-learn](https://scikit-learn.org/stable/)
*  [matplotlib](https://matplotlib.org/)

## Get in Touch

_If you have any questions not covered in this overview, please contact the GeminiMol team at wanglin3@shanghaitech.edu.cn. We would love to hear your feedback and understand how GeminiMol has been useful in your research. Share your stories with us at wanglin3@shanghaitech.edu.cn or baifang@shanghaitech.edu.cn._  

