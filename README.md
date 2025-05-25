# ADNI-FreeSurfer-Segmentation-Machine-Learning
## MRI Radiomics Processing Pipeline

This repository contains tools for processing MRI data and extracting radiomics features.

### Installation

#### Prerequisites
- Python 3.7+
- pip
- Git (only needed if direct installation fails)

#### 1. Install PyRadiomics

First, ensure your Python environment is up to date:

```bash
pip install --upgrade pip setuptools wheel
pip install pyradiomics==3.0.1
```

If installation fails, try building from source:
```bash
git clone https://github.com/AIM-Harvard/pyradiomics.git
cd pyradiomics
pip install .
```

Install aditional requirements:
```bash
pip install nibabel SimpleITK nilearn pandas numpy
```

## Patient Selection

For this project, the initial step involved requesting access to the Alzheimer's Disease Neuroimaging Initiative (ADNI) database. The focus was placed on patients who were initially diagnosed with Mild Cognitive Impairment (MCI), specifically identifying those who progressed to Alzheimer's Disease (AD) within a 24-month time frame. This allowed the construction of a dataset comprising two groups: MCI converters (patients who developed AD within 24 months) and MCI non-converters (patients who remained stable during the same period).

### Dataset
https://www.kaggle.com/datasets/joeldv1/adni-change-mci-to-alzheimer-in-24-months

### References
Danvalcor, jcg01, DairaRe. (2024). ADNI-FreeSurfer-Segmentation. Retrieved from: https://github.com/Danvalcor/ADNI-FreeSurfer-Segmentation
