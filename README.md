# 🧠 From Promise to Practical Reality: Transforming Diffusion MRI Analysis with Fast Deep Learning Enhancement

> [Xinyi Wang](https://scholar.google.com/citations?user=_uPPBqUAAAAJ&hl=en)  
> [Michael Barnett](https://scholar.google.com.au/citations?user=iZVWDzwAAAAJ&hl=en)  
> [Frederique Boonstra](https://scholar.google.com.au/citations?user=xHxerDoAAAAJ&hl=en)  
> [Yael Barnett](https://scholar.google.com.au/citations?hl=en&user=TVSkAYsAAAAJ)  
> [Mariano Cabezas](https://scholar.google.com.au/citations?hl=en&user=zPs-kAkAAAAJ)  
> [Arkiev D'Souza](https://scholar.google.com.au/citations?user=eqO2au8AAAAJ&hl=en)  
> [Matthew C. Kiernan](https://scholar.google.com.au/citations?user=J7M4CGMAAAAJ&hl=en)  
> Kain Kyle  
> [Meng Law](https://scholar.google.com.au/citations?user=lKi-yTMAAAAJ&hl=en)  
> Lynette Masters  
> [Zihao Tang](https://scholar.google.com.au/citations?user=JAfD8moAAAAJ&hl=en)  
> [Stephen Tisch](https://scholar.google.com.au/citations?hl=en&user=hnKJB1YAAAAJ)  
> [Sicong Tu](https://scholar.google.com.au/citations?hl=en&user=z44EzHAAAAAJ)  
> [Anneke Van Der Walt](https://scholar.google.com.au/citations?hl=en&user=F3AJeqQAAAAJ)  
> [Dongang Wang](https://scholar.google.com.au/citations?hl=en&user=Rs7zEZoAAAAJ)  
> [Fernando Calamante](https://scholar.google.com.au/citations?user=_6_n0PIAAAAJ&hl=en)\*  
> [Weidong Cai](https://scholar.google.com.au/citations?hl=en&user=N8qTc2AAAAAJ)\*  
> [Chenyu Wang](https://scholar.google.com.au/citations?user=mo0AoZAAAAAJ&hl=en)\*  

> \* These authors contributed equally as senior authors.

---

### 🔬 Abstract

Fiber orientation distribution (FOD) models are vital for analyzing complex white matter fiber configurations and supporting tractography/connectome pipelines. Clinical use, however, is limited due to quality constraints in routine diffusion MRI.

We present **FastFOD-Net**, a fast, end-to-end deep learning framework for FOD enhancement. It delivers high accuracy across **six neurological disorders**, with **60× speedup** over its predecessor. This work demonstrates the framework's clinical viability and encourages trust in deep learning-enhanced diffusion MRI for real-world data.

> [[🌐 Project Page]](https://fastfodnet.github.io/)

<p align="center">
  <img src="./teaser.png" alt="FastFOD-Net Teaser" width="80%">
</p>

---

> [[Project page]](https://fastfodnet.github.io/)
>
> ![FastFOD-Net](./teaser.png)

## Outline
1. [Introduction](#fastfod-net)  
2. [Data Preprocessing](#dataprocessing)  
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

---
## Data Preprocessing
1. [DWI Denoising](https://mrtrix.readthedocs.io/en/latest/dwi_preprocessing/denoising.html)
2. [DWI Distortion Correction](https://mrtrix.readthedocs.io/en/latest/dwi_preprocessing/dwifslpreproc.html)
3. FOD Generation
   - [Response Function Estimation](https://mrtrix.readthedocs.io/en/latest/constrained_spherical_deconvolution/response_function_estimation.html)
   - [SS3T CSD](https://mrtrix.readthedocs.io/en/latest/constrained_spherical_deconvolution/constrained_spherical_deconvolution.html)
   - [MSMT CSD](https://mrtrix.readthedocs.io/en/latest/constrained_spherical_deconvolution/multi_shell_multi_tissue_csd.html)

---
## 📘 Description of Arguments
| Argument                     | Description                                                    |
| ---------------------------- | -------------------------------------------------------------- |
| `--dataroot`                 | Path to low-resolution FOD input data.                         |
| `--maskroot`                 | Path to brain mask files.                                      |
| `--gtroot`                   | Path to high-resolution ground truth FODs.                     |
| `--checkpoints_dir`          | Directory to save model checkpoints.                           |
| `--name`                     | Experiment name, used to label checkpoint folder.              |
| `--normalization_mode`       | Normalization strategy (e.g., z-score based).                  |
| `--model`                    | Model type; `re` refers to resolution enhancement.             |
| `--input_nc` / `--output_nc` | Number of channels (e.g., 45 SH coefficients).                 |
| `--init_type`                | Weight initialization method (`kaiming`, etc).                 |
| `--dataset_mode`             | Dataset loader mode (`fod_re` for FOD resolution enhancement). |
| `--num_threads`              | Number of data loading threads.                                |
| `--batch_size`               | Batch size for training.                                       |
| `--beta1`                    | Beta1 for Adam optimizer.                                      |
| `--lr`                       | Initial learning rate.                                         |
| `--n_epochs`                 | Number of training epochs.                                     |
| `--print_freq`               | Print frequency (in iterations).                               |
| `--save_latest_freq`         | Frequency (in iterations) to save latest checkpoint.           |
| `--save_epoch_freq`          | Frequency (in epochs) to save checkpoints.                     |
| `--gpu_ids`                  | GPU to use (e.g., `0` for first GPU).                          |
| `--conv_type`                | Network architecture; here using `fastfodnet`.                 |
| `--test_fold`                | Index of fold used for testing (for cross-validation).         |
| `--phase`                    | Run phase: `train` or `splitfolds` (for k-fold split).         |
| `--index_pattern`            | Regex pattern to match subject IDs.                            |
| `--sample_suffix`            | File suffix for low-res FODs.                                  |
| `--sample_gt_suffix`         | File suffix for ground truth FODs.                             |
| `--foldroot`                 | Directory containing train/test splits.                        |

---

## Basic Usages
### 🏋️‍♂️ Training
To start training FastFOD-Net using predefined parameters (e.g., for the MSBIR dataset), run:
```
cd ./CORE/scripts/
sh train_msbir.sh
```
- This script wraps a python train_model.py command with the appropriate dataset, fold, and model settings.
- 📝 Be sure to modify `train_msbir.sh` if you're working with a different dataset or configuration.

### 🔍 Inference
To perform inference on a trained model:
```
cd /scripts/
sh test_msbir.sh
```
- `test.sh` should call test_model.py or equivalent with correct model checkpoint and dataset paths.
- 📝 Ensure the --phase is set to test or similar.

---
## 🧪 Evaluation Pipeline

This pipeline provides tools for evaluating the performance of Fibre Orientation Distribution enhancement methods using a set of quantitative metrics from different perspectives.

---
### 📈 FOD Evaluation
The following metrics are computed in `evaluation_fod.py`:
- **MSE**: Mean Squared Error   
- **PSNR**: Peak Signal-to-Noise Ratio  
- **$r_{\text{Angular}}$**: Angular correlation coefficient
  
#### 🔧 Example Usage
To run FOD evaluation:
```
cd ./evaluation
python run_fod_metrics.py
```
---
### 🧠 Fiber Bundle Element ("Fixel") Evaluation

This module evaluates fiber-specific metrics derived from fixel-based analysis.

#### 🛠 Generate Fixels

To generate fixel data from FOD images:
```
cd ./evaluation
python generate_fixel.py
```

#### 🛠 Generate ROIs

Bundle-wise ROIs can be generated using [TractSeg](https://github.com/MIC-DKFZ/TractSeg), a tool for white matter tract segmentation.

We use specific bundles to define regions of different fiber complexity:

- **Single-fiber region:**  
  - `CC` (Corpus Callosum)

- **Two-crossing-fiber regions:**  
  - `MCP` (Middle Cerebellar Peduncle)  
  - `CST` (Corticospinal Tract)

- **Three-crossing-fiber regions:**  
  - `SLF` (Superior Longitudinal Fasciculus)  
  - `CST`  
  - `CC`
 
#### 🛠 Generate ROIs

Bundle-wise ROIs can be generated using [TractSeg](https://github.com/MIC-DKFZ/TractSeg), a tool for white matter tract segmentation.

We define regions with different fiber complexities using the following bundles:

- **Single-fiber region:**  
  - `CC` (Corpus Callosum)

- **Two-crossing-fiber regions:**  
  - `MCP` (Middle Cerebellar Peduncle)  
  - `CST` (Corticospinal Tract)

- **Three-crossing-fiber regions:**  
  - `SLF` (Superior Longitudinal Fasciculus)  
  - `CST`  
  - `CC`

🔍 *See* `./evaluation/generate_fixel_roi.py` for an example of how to generate these ROIs using results from TractSeg.*


#### 📊 Fixel Evaluation Metrics

This module performs evaluation on fixel-based metrics after fixel-wise matching between methods. Implemented in `evaluation_fixel.py`, the following metrics are computed:

- **$E_{Angular}$**: Angular error between matched fixels  
- **$E_{FD}$**: Error in Fixel Density (FD)  
- **$E_{Peak}$**: Error in Peak

#### 🔧 Example Usage

Run the full fixel evaluation pipeline:
```
cd ./evaluation
python run_fixel_metrics.py
```
---
### Connectome evaluation

#### 🛠 [Structure Connectome Construction](https://mrtrix.readthedocs.io/en/latest/quantitative_structural_connectivity/structural_connectome.html)
1. Get a parcellation image with [FastSurfer](https://github.com/Deep-MI/FastSurfer) using [Desikan-Killiany Atlas 84](file:///Users/xinyiwang/Downloads/jnnp-2021-328185-inline-supplementary-material-1.pdf)
2. [Anatomically-Constrained Tractography (ACT)](https://mrtrix.readthedocs.io/en/latest/quantitative_structural_connectivity/act.html)
3. [Spherical-deconvolution Informed Filtering of Tractograms (SIFT)](https://mrtrix.readthedocs.io/en/latest/quantitative_structural_connectivity/sift.html)

#### 🛠 Connectome Metrics

We evaluate structural connectomes using the following metrics:

- **Disparity**: Quantifies variability in edge weights within the connectome.
- **Number of significantly different edges**: Counts edges with statistically significant differences across methods.

📁 See implementation: [`./evaluation/connectome.py`](./evaluation/connectome.py)

#### 🛠 Graph Metrics

We treat the connectome as a graph and compute higher-order network properties inspired by:

- **Brain Connectivity Toolbox (BCT)**  
  - Website: [https://sites.google.com/site/bctnet/](https://sites.google.com/site/bctnet/)
- Python implementation: [`./evaluation/graph_metrics.py`](./evaluation/graph_metrics.py)

#### 📈 Example Usage

To run the full pipeline of connectome and graph metric evaluation:

```bash
cd ./evaluation
python run_connectome_metrics.py
```


### Fixel-based analysis
check if significant differences between patients and controls can be preseved after useing deep leanring 
1. [Fixel-based analysis for MSMT CSD](https://mrtrix.readthedocs.io/en/latest/fixel_based_analysis/st_fibre_density_cross-section.html)
2. Fixel matching in [`./evaluation/evaluaation_fixel.py`](./evaluation/graph_metrics.py) between methods for comparision
3. Stats


### Pathological connection analysis
TBC

### Correlation analysis
TBC

## Citation
If you find our data or project useful in your research, please cite:

```
TBC
@inproceedings{
}
```
#### Acknowledgments
This repo. template was borrowed from [Chaoyi Zhang's Project](https://github.com/chaoyivision/SGGpoint). 

