<div align="center">

<h2>MRI Contrast Enhancement Kinetics World Model</h2>

CVPR2026

#### [Jindi Kong<sup>1</sup>](https://scholar.google.com/citations?hl=zh-CN&tzom=300&user=Kr-XKIwAAAAJ) · [Yuting He<sup>1</sup>](https://yutinghe-list.github.io/) · Cong Xia<sup>2</sup> · [Rongjun Ge<sup>3</sup>](https://scholar.google.com/citations?user=v8K8HIkAAAAJ&hl=zh-CN) · [Shuo Li<sup>1</sup>](https://scholar.google.com/citations?hl=en&user=6WNtJa0AAAAJ&view_op=list_works&sortby=pubdate) 
<sup>1</sup> Case Western Reserve University, Cleveland, OH, USA  &nbsp;&nbsp; <sup>2</sup> Jiangsu Cancer Hospital, Nanjing, Jiangsu, China  &nbsp;&nbsp; <sup>3</sup> Southeast University, Nanjing, Jiangsu, China

<a href='https://arxiv.org/pdf/2602.19285'><img src='https://img.shields.io/badge/arXiv-Paper-ffc107'></a>

<img src="figs/1.gif" alt="highlight" width="23%">
<img src="figs/2.gif" alt="highlight" width="23%">
<img src="figs/3.gif" alt="highlight" width="23%">
<img src="figs/4.gif" alt="highlight" width="23%">

</div>



## ⚙️ The 1st MRI CEKWorld

<div align="center">
  <img src="figs/world model_version.png" alt="Image 1" width="50%">
</div>

**(a) Task** Our MRI Contrast Enhancement Kinetics World Model (MRI CEKWorld) generates contrast-enhanced sequences that conform to kinetics in the human body after contrast agent injection. 
**(b) Problem** Clinical contrast MRI acquisition presents inefficient information yield with adverse risks and higher cost, but a fixed, sparse sequence.
**(c) Adavantages** Our MRI CEKWorld enables continuous contrast-free dynamics with no contrast agent risks, low cost, and convenience



## 🚀 Getting Started
#### 🛠 Installation
Following the ControlNet installation pipeline, the environment configuration can be found [here](https://github.com/lllyasviel/ControlNet/blob/main/environment.yaml):

```bash
conda env create -f environment.yaml
conda activate control
```

#### ⏬ Download weights

The model is trained based on [ControlNet-v1-1](https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_canny.pth) and transferred using [tool_transfer_control](https://github.com/lllyasviel/ControlNet/blob/main/tool_transfer_control.py). Alternatively, you can find the processed initial weights here: [Google Drive](https://drive.google.com/file/d/1V0VVO9YhB_Xh5RG9rmJ8FcRKTnTLS9or/view?usp=drive_link) 🔗.


Our final weights, optimized for abdominal datasets, can be downloaded from the provided links [Google Drive](https://drive.google.com/file/d/1klM0pbL_AT2V0HMm5drbAAoxcQsgPHro/view?usp=drive_link)🔗.

#### 🛠️ Framework
<div align="center">
  <img src="figs/method3_1.png" alt="Image 1" width="80%">
</div>


##### Where to find the LAL and LDL

Latent Alignment Learning (LAL) is implemented in `ldm/models/diffusion/ddpm.py`.  
The spatial alignment losses are computed in:
- `compute_log_cholesky_template_and_losses(...)`

This function contains the code for building the latent template (log-Cholesky parameterization) and computing the corresponding spatial consistency/alignment losses.

Latent Difference Learning (LDL) is implemented in `ldm/models/diffusion/ddpm.py`.  
The temporal difference (smoothness) losses are computed in:
- `second_order_temporal_loss_batch(...)`

This function implements the temporal difference regularization used to encourage smooth dynamics over time. The $K_i$ (`num_samples_per_time_interval`) is recommended to try starting from 0 and gradually increasing. If excellent spatial alignment is not achieved, the generated results may collapse.


##### Dataset sampling
The dataset is sampled using the `PatientSliceBatchSampler`, ensuring that each batch contains data from the same slice of the same patient, but at different time points.

## ⭐ Citation
If you find MRI CEKWorld useful for your research, welcome to cite our work using the following BibTeX:
```bibtex
@article{kong2026mri,
  title={MRI Contrast Enhancement Kinetics World Model},
  author={Kong, Jindi and He, Yuting and Xia, Cong and Ge, Rongjun and Li, Shuo},
  journal={arXiv preprint arXiv:2602.19285},
  year={2026}
}
```

## ⏬ The preprocessed Duke DCE-MRI

The 2D tumor patches provided in this dataset were derived from the **DUKE-BREAST-CANCER-MRI** collection hosted by The Cancer Imaging Archive (TCIA):

Saha, A., Harowicz, M. R., Grimm, L. J., Weng, J., Cain, E. H., Kim, C. E., Ghate, S. V., Walsh, R., & Mazurowski, M. A. (2021). *Dynamic contrast-enhanced magnetic resonance images of breast cancer patients with tumor locations* [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.e3sv-re93

The original imaging data are licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

To generate this derived dataset, tumor-containing MRI slices were selected from the original images. Regions containing the tumors were cropped, resized to a standardized image size, and converted into NumPy (`.npy`) arrays.

The resulting 2D tumor patches contain image-derived material from the original TCIA dataset and therefore remain subject to the applicable **CC BY-NC 4.0** license terms. These image-derived data may be shared and adapted for non-commercial purposes subject to appropriate attribution.

Users should cite the original DUKE-BREAST-CANCER-MRI dataset and comply with the applicable TCIA Data Usage Policies and Restrictions.


https://drive.google.com/file/d/1bV3pEe5O-bqKaDyLU5hqRVwodnCoyre4/view?usp=sharing

## ❤️ Acknowledgement
This code is mainly built upon [ControlNet](https://github.com/lllyasviel/ControlNet/tree/main),  thanks to their invaluable contributions.
