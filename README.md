# From Promise to Practical Reality: Transforming Diffusion MRI Analysis with Fast Deep Learning Enhancement


>
> [Xinyi Wang](https://scholar.google.com/citations?user=_uPPBqUAAAAJ&hl=en), [Michael Barnett](https://scholar.google.com.au/citations?user=iZVWDzwAAAAJ&hl=en), [Frederique Boonstra](https://scholar.google.com.au/citations?user=xHxerDoAAAAJ&hl=en&oi=ao), [Yael Barnett](https://scholar.google.com.au/citations?hl=en&user=TVSkAYsAAAAJ), [Mariano Cabezas](https://scholar.google.com.au/citations?hl=en&user=zPs-kAkAAAAJ&view_op=list_works&sortby=pubdate), [Arkiev D'Souza](https://scholar.google.com.au/citations?user=eqO2au8AAAAJ&hl=en&oi=ao), [Matthew C. Kiernan](https://scholar.google.com.au/citations?user=J7M4CGMAAAAJ&hl=en&oi=ao), Kain Kyle, [Meng Law](https://scholar.google.com.au/citations?user=lKi-yTMAAAAJ&hl=en&oi=ao), Lynette Masters, [Zihao Tang](https://scholar.google.com.au/citations?user=JAfD8moAAAAJ&hl=en&oi=ao), [Stephen Tisch](https://scholar.google.com.au/citations?hl=en&user=hnKJB1YAAAAJ), [Sicong Tu](https://scholar.google.com.au/citations?hl=en&user=z44EzHAAAAAJ), [Anneke Van Der Walt](https://scholar.google.com.au/citations?hl=en&user=F3AJeqQAAAAJ), [Dongang Wang](https://scholar.google.com.au/citations?hl=en&user=Rs7zEZoAAAAJ), [Fernando Calamante](https://scholar.google.com.au/citations?user=_6_n0PIAAAAJ&hl=en&oi=ao)\*, [Weidong Cai](https://scholar.google.com.au/citations?hl=en&user=N8qTc2AAAAAJ)\*, [Chenyu Wang](https://scholar.google.com.au/citations?user=mo0AoZAAAAAJ&hl=en&oi=ao)\*
>
\* These authors contributed equally as senior authors.
>
>
> **Abstract** 
> Fiber orientation distribution (FOD) is an advanced diffusion MRI modeling technique that represents complex white matter fiber configurations, and a key step for subsequent brain tractography and connectome analysis. Its reliability and accuracy, however, heavily rely on the quality of the MRI acquisition and the subsequent estimation of the FODs at each voxel. Generating reliable FODs from widely available clinical protocols with single-shell and low-angular-resolution acquisitions remains challenging but could potentially be addressed with recent advances in deep learning-based enhancement techniques. Despite advancements, existing methods have predominantly been assessed on healthy subjects, which have proved to be a major hurdle for their clinical adoption. In this work, we validate a newly optimized enhancement framework, FastFOD-Net, across healthy controls and six neurological disorders. 
This accelerated end-to-end deep learning framework enhancing FODs with superior performance and delivering training/inference efficiency for clinical use ($60\times$ faster comparing to its predecessor).
With the most comprehensive clinical evaluation to date, our work demonstrates the potential of FastFOD-Net in accelerating clinical neuroscience research, empowering diffusion MRI analysis for disease differentiation, improving interpretability in connectome applications, and reducing measurement errors to lower sample size requirements. Critically, this work will facilitate the more widespread adoption of, and build clinical trust in, deep learning based methods for diffusion MRI enhancement.  Specifically, FastFOD-Net enables robust analysis of real-world, clinical diffusion MRI data, comparable to that achievable with high-quality research acquisitions.
<!-- > ![image](https://user-images.githubusercontent.com/39485479/227234185-074da035-f4f9-4e12-bd3e-e5070167ba74.png) -->
>
> 

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

   
## Data Preprocessing
1. [DWI Denoising](https://mrtrix.readthedocs.io/en/latest/dwi_preprocessing/denoising.html)
2. [DWI Distortion Correction](https://mrtrix.readthedocs.io/en/latest/dwi_preprocessing/dwifslpreproc.html)
3. FOD Generation
   - [Response Function Estimation](https://mrtrix.readthedocs.io/en/latest/constrained_spherical_deconvolution/response_function_estimation.html)
   - [SS3T CSD](https://mrtrix.readthedocs.io/en/latest/constrained_spherical_deconvolution/constrained_spherical_deconvolution.html)
   - [MSMT CSD](https://mrtrix.readthedocs.io/en/latest/constrained_spherical_deconvolution/multi_shell_multi_tissue_csd.html)

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


## 🧪 Evaluation Pipeline

This pipeline provides tools for evaluating the performance of Fibre Orientation Distribution enhancement methods using a set of quantitative metrics from different perspectives.


### FOD Evaluation
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

