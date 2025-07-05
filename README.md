# FastFOD-Net


This repository contains the official implementation of the following paper:

>
From Promise to Practical Reality: Transforming Diffusion MRI Analysis with Fast Deep Learning Enhancement
>
> [Xinyi Wang](https://scholar.google.com/citations?user=_uPPBqUAAAAJ&hl=en), [Michael Barnett](https://scholar.google.com.au/citations?user=iZVWDzwAAAAJ&hl=en), [Frederique Boonstra](https://scholar.google.com.au/citations?user=xHxerDoAAAAJ&hl=en&oi=ao), [Yael Barnett](https://scholar.google.com.au/citations?hl=en&user=TVSkAYsAAAAJ), [Mariano Cabezas](https://scholar.google.com.au/citations?hl=en&user=zPs-kAkAAAAJ&view_op=list_works&sortby=pubdate), [Arkiev D'Souza](https://scholar.google.com.au/citations?user=eqO2au8AAAAJ&hl=en&oi=ao), [Matthew C. Kiernan](https://scholar.google.com.au/citations?user=J7M4CGMAAAAJ&hl=en&oi=ao), [Kain Kyle](), [Meng Law](https://scholar.google.com.au/citations?user=lKi-yTMAAAAJ&hl=en&oi=ao), [Lynette Masters](), [Zihao Tang](https://scholar.google.com.au/citations?user=JAfD8moAAAAJ&hl=en&oi=ao), [Stephen Tisch](https://scholar.google.com.au/citations?hl=en&user=hnKJB1YAAAAJ), [Sicong Tu](https://scholar.google.com.au/citations?hl=en&user=z44EzHAAAAAJ), [Anneke Van Der Walt](https://scholar.google.com.au/citations?hl=en&user=F3AJeqQAAAAJ), [Dongang Wang](https://scholar.google.com.au/citations?hl=en&user=Rs7zEZoAAAAJ), [Fernando Calamante](https://scholar.google.com.au/citations?user=_6_n0PIAAAAJ&hl=en&oi=ao)*, [Weidong Cai](https://scholar.google.com.au/citations?hl=en&user=N8qTc2AAAAAJ)*, [Chenyu Wang](https://scholar.google.com.au/citations?user=mo0AoZAAAAAJ&hl=en&oi=ao)*
>**\*\*** These authors contributed equally as senior authors.
>
>
> **Abstract** 
> Modern structural brain connectome pipelines and tractography techniques heavily rely on the quality of the diffusion weighted image acquisition (and angular resolution in particular) and the subsequent estimation of the fiber orientation distributions (FODs) for each voxel. Generating reliable connectomes from low angular single shell acquisitions in clinical scenarios remains a challenging task. This work presents an end-to-end deep learning framework to enhance FOD estimates according to multi shell acquisitions from low angular single shell acquisitions to guarantee high quality tractography and connectomes within acceptable time and resources.
<!-- > ![image](https://user-images.githubusercontent.com/39485479/227234185-074da035-f4f9-4e12-bd3e-e5070167ba74.png) -->
>
> 

> [[Project page]](https://fastfodnet.github.io/)
>
> ![FastFOD-Net](./teaser.png)

## Outline
1. [Introduction](#fastfod-net)  
2. [Dataset](#dataset)  
3. [Usage](#usage)  
   - [Training](#training)  
   - [Inference](#inference)  
4. [Evaluation](#evaluation)  
   - [FOD Evaluation](#fod-evaluation)  
   - [Fiber Bundle Element "fixel" Evaluation](#fiber-bundle-element-fixel-evaluation)  
   - [Connectome Evaluation](#connectome-evaluation)  
   - [Fixel-Based Analysis](#fixel-based-analysis)  
   - [Pathological Connection Analysis](#pathological-connection-analysis)  
   - [Correlation Analysis](#correlation-analysis)  
4. [References](#references)  

   
## Dataset
[HCP Dataset](https://www.humanconnectome.org/)

## Basic Usages
### **Training**
```
cd /scripts/
sh train.sh
```

### **Inference**
```
cd /scripts/
sh test.sh
```


## Evaluation Pipeline
TBC

### FOD evaluation
```
cd /evaluation
python evaluation_fod.py
```
### Fiber Bundle Element "fixel" Evaluation
- Generate fixel
```
cd /evaluation
python generate_fixel.py
```
- Evaluate fixels from direction, peak, AFD
```
cd /evaluation
python evaluate_fixel.py
```
### Connectome evaluation
### Fixel-based analysis
### Pathological connection analysis
### Correlation analysis


## Citation

If you find our data or project useful in your research, please cite:

```
TBC

@inproceedings{
}
```
#### Acknowledgments
This repo. template was borrowed from [Chaoyi Zhang's Project](https://github.com/chaoyivision/SGGpoint). 

